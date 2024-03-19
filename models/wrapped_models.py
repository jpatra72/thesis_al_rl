import sys
import copy
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tools.helper import DEVICE
from pbnn.src.pbnn.pnn import ProgressiveNeuralNetwork


class BaseNetworkWrapper(ABC):
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss(),
                 weight_decay: float = 0,
                 learning_rate: float = 2e-3,
                 num_epochs: int = 20,
                 patience: int = 5,
                 tqdm_bar_enable: bool = False):
        self.model = model
        self.criterion = criterion
        self.model.to(DEVICE)
        self.criterion.to(DEVICE)
        self.num_epochs = num_epochs
        self.patience = patience

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params=self.params,
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        self.reduced_epoch_factor: Optional[int] = None
        self.epoch_queried_count_threshold: Optional[int] = None

        self.tqdm_bar_enable = tqdm_bar_enable

        self.best_model_state_dict = None
        self.init_state_dict = self.get_state_dict()

    def _model_reset(self):
        self.best_model_state_dict = None

    def _run_epoch(self,
                   dataloader: DataLoader,
                   tqdm_prefix: Optional[str] = None):
        self.model.train()
        average_loss = 0
        with tqdm(dataloader, file=sys.stdout, disable=not self.tqdm_bar_enable) as t:
            t.set_description(tqdm_prefix)
            for features, targets in t:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                logits = self.model(features)

                if not isinstance(logits, list):
                    logits = [logits]

                logit = logits[-1]
                loss = self.criterion(logit, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                average_loss += loss.detach() / len(dataloader)
                t.set_postfix(loss=loss.detach().item())
        return average_loss

    def set_values_from_env(self,
                            reduced_epoch_factor: Optional[float] = None,
                            epoch_queried_count_threshold: Optional[int] = None):
        self.reduced_epoch_factor = reduced_epoch_factor
        self.epoch_queried_count_threshold = epoch_queried_count_threshold

    def train_model(self,
                    dataloader: DataLoader,
                    queried_count: Optional[int] = None):
        num_epochs = self.num_epochs
        # current_loss = 0
        # best_loss = float('Inf')
        if self.reduced_epoch_factor and self.epoch_queried_count_threshold:
            assert queried_count is not None
            num_epochs = self.num_epochs if queried_count > self.epoch_queried_count_threshold \
                else int(self.num_epochs * self.reduced_epoch_factor)

        prefix = None
        self.lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.5, verbose=True, patience=5)
        for epoch in range(num_epochs):
            if self.tqdm_bar_enable:
                str_len = str(len(str(epoch)))
                prefix = ('Epoch {epoch:' + str_len + 'd}: Train').format(epoch=epoch+1)
            loss = self._run_epoch(dataloader=dataloader, tqdm_prefix=prefix)
            # self.lr_schedulers.step(loss)
        #     current_loss = self._run_epoch(dataloader=dataloader, tqdm_prefix=prefix)
        #     if current_loss < best_loss:
        #         best_loss = current_loss
        #         best_model_state_dict = self.get_state_dict()
        #         steps_without_improvement = 0
        #     else:
        #         steps_without_improvement += 1
        #         if steps_without_improvement >= self.patience:
        #             break
        # self.set_state_dict(best_model_state_dict)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss

    def get_predictions(self, features: torch.Tensor) \
            -> torch.Tensor:
        features = features.to(DEVICE)
        logits = self.model(features)
        if not isinstance(logits, list):
            logits = [logits]
        logit = logits[-1]
        return logit

    def get_probabilities(self, features: torch.Tensor) \
            -> Tuple[np.ndarray, np.ndarray]:
        '''
        :param features:
        :return: model predictions and metric for model confidence
        '''
        self.model.eval()
        with torch.no_grad():
            logit = self.get_predictions(features)
        prob, entropy = self.get_entropy(logit)
        return prob.cpu().numpy(), entropy.cpu().numpy()

    def test_score(self, dataloader: DataLoader) \
            -> (float, float):
        correct = 0
        total = 0
        avg_entropy = 0
        # self.entropy_list = []
        self.model.eval()
        fs, ls = [], []
        with torch.no_grad():
            for features, targets in dataloader:
                if isinstance(targets, tuple):
                    targets = targets[1]
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                logit = self.get_predictions(features)
                _, entropy = self.get_entropy(logit)
                _, pred = torch.max(logit, dim=1)
                total += targets.shape[0]
                correct += (pred == targets).sum().item()
                avg_entropy += entropy.sum().item()
                # self.entropy_list.extend(entropy.cpu().numpy())
            #     fs.append(features)
            #     ls.append(targets)
            # self.f_tensor = torch.concatenate(fs, dim=0)
            # self.l_tensor = torch.concatenate(ls, dim=0)
            # arr = np.array(self.entropy_list)
            # arr_indices = np.where(arr < 0.15)[0]
            # self.f_tensor = self.f_tensor[arr_indices]
            # self.l_tensor = self.l_tensor[arr_indices]
            # logits = self.get_predictions(self.f_tensor)
            # _, preds = torch.max(logits, dim=1)
            # self.acc = (preds == self.l_tensor).sum().item() / len(self.l_tensor)

        accuracy = correct / total
        avg_entropy /= total
        # if self.tqdm_bar_enable:
        #     print(f'Test: accuracy: {accuracy:.4f}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return accuracy, avg_entropy

    def get_entropy(self, logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        logprob = self.log_softmax(logits)
        prob = torch.exp(logprob)
        p_log_p = prob * logprob
        entropy = -p_log_p.sum(dim=1)
        return prob, entropy

    @abstractmethod
    def reset_model(self, reset_random: Optional[bool] = None):
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def set_state_dict(self, state_dict):
        raise NotImplementedError


class WrappedLenet5(BaseNetworkWrapper):
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: Callable[[Tensor, Tensor], Tensor],
                 weight_decay: float = 0,
                 learning_rate: float = 2e-3,
                 num_epochs: int = 20,
                 patience: int = 5,
                 tqdm_bar_enable: bool = False):
        super().__init__(model=model,
                         criterion=criterion,
                         weight_decay=weight_decay,
                         learning_rate=learning_rate,
                         num_epochs=num_epochs,
                         patience=patience,
                         tqdm_bar_enable=tqdm_bar_enable)

    def reset_model(self, reset_random=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.load_state_dict(self.init_state_dict)

    def get_state_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class WrappedPNN(BaseNetworkWrapper):
    def __init__(self,
                 model: ProgressiveNeuralNetwork,
                 criterion: Callable[[Tensor, Tensor], Tensor],
                 weight_decay: float = 0,
                 learning_rate: float = 2e-3,
                 num_epochs: int = 20,
                 patience: int = 5,
                 tqdm_bar_enable: bool = False):
        super().__init__(model=model,
                         criterion=criterion,
                         weight_decay=weight_decay,
                         learning_rate=learning_rate,
                         num_epochs=num_epochs,
                         patience=patience,
                         tqdm_bar_enable=tqdm_bar_enable)


    def reset_model(self, reset_random: Optional[bool] = None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.networks.load_state_dict(self.init_state_dict)

    def get_state_dict(self):
        return copy.deepcopy(self.model.networks.state_dict())

    def set_state_dict(self, state_dict):
        self.model.network.load_state_dict(state_dict)


if __name__ == "__main__":
    from torchvision import transforms
    from torchvision.datasets import MNIST

    from tools.helper import torch_data_path
    from models import lenet5

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    dataset_train = MNIST(root=torch_data_path, train=True, download=True,
                            transform=transform)
    dataset_test = MNIST(root=torch_data_path, train=False, download=True,
                           transform=transform)

    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=0)

    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=0)

    # test wrapped lenet clas
    lenet5 = lenet5()
    wrapped_lenet5 = WrappedLenet5(model=lenet5,
                                   criterion=torch.nn.CrossEntropyLoss(),
                                   num_epochs=2,
                                   tqdm_bar_enable=True)
    wrapped_lenet5.train_model(dataloader_train)
    accuracy = wrapped_lenet5.test_score(dataloader_test)

    # test wrapped pnn class
    pnn = ProgressiveNeuralNetwork(
        base_network=lenet5,
        backbone=None,
        last_layer_name='11',
        lateral_connections=['3', '7', '9']
    )
    pnn.add_new_column(is_classification=True, output_size=len(dataset_train.classes))
    wrapped_pnn = WrappedPNN(model=pnn,
                             criterion=torch.nn.CrossEntropyLoss(),
                             num_epochs=2,
                             tqdm_bar_enable=True)
    wrapped_pnn.train_model(dataloader_train)
    accuracy = wrapped_pnn.test_score(dataloader_test)