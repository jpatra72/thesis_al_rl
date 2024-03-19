import copy
import time
from typing import Optional, Dict, Union

import gymnasium as gym
import custom_envs
from gymnasium import spaces

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset

from models.wrapped_models import BaseNetworkWrapper

from tools.mnist_dataloaders import get_dataloaders, get_balanced_datasets, IndexedSubset
from tools.helper import DEVICE


# TODO: use pin_memory?
class ImageClassifier_new(gym.Env):

    def __init__(self,
                 model: BaseNetworkWrapper,
                 dataset: Union[Dataset, Subset],
                 test_dataset: Union[Dataset, Subset],
                 al_budget: int,
                 state_type: str = 'prob_preds',
                 labelled_samples_per_class: int = 20,
                 reward_samples_per_class: int = 50,
                 model_kwargs: Optional[Dict] = None,
                 eval_env: bool = False,
                 verbose: bool = False):
        self.classifier_model = model
        if model_kwargs:
            self.classifier_model.set_values_from_env(**model_kwargs)
        self.al_budget = al_budget
        self.eval_env = eval_env
        self.state_type = state_type
        self.labelled_samples_per_class = labelled_samples_per_class
        self.reward_samples_per_class = reward_samples_per_class

        self.dataset = dataset
        self.test_dataset = test_dataset
        self.dataset_reward, self.dataset_init = get_balanced_datasets(self.dataset, samples_per_class=self.reward_samples_per_class)

        self.dataloader_reward = DataLoader(self.dataset_reward, batch_size=512, shuffle=False)
        self.dataloader_test = DataLoader(self.test_dataset, batch_size=512, shuffle=False)

        self.dataset_complete_normed = (self.dataset.dataset.tensors[0] /
                                        torch.linalg.norm(self.dataset.dataset.tensors[0], ord=2, dim=1).unsqueeze(1))
        self.dataset_complete_normed = self.dataset_complete_normed.to(DEVICE)

        self._create_dataloaders()

        self.verbose = verbose
        self.init_labelled_count = len(self.dataset_labelled)
        self.queried_count = 0
        self.step_count = 0
        self.action_count = 0
        self.performance = 0.0
        self._entropy_test = None
        self._entropy_test_eps = []

        self.image_shape = next(iter(self.dataloader_learn))[0].shape  # shape = (1, n_channels, n_height, n_width)
        self.class_count = 10

        # Observations are dictionaries containing the image, model's class predictions and confidence.
        # TODO: modify to parameters instead of values
        if self.state_type == 'sr1_womargin' or self.state_type == 'sr2_womargin':
            full_shape = self.class_count + 1
        else:
            full_shape = self.class_count + 2
        self.observation_space = spaces.Box(float('-inf'), float('inf'), shape=(full_shape,), dtype=np.float32)

        # We have 2 actions, corresponding to 0, 1 corresponding to do not query label and query label respectively
        self.action_space = spaces.Discrete(2)

    def _create_dataloaders(self):

        self.dataset_labelled, self.dataset_learn = get_balanced_datasets(self.dataset_init, samples_per_class=self.labelled_samples_per_class, shuffle=True)
        self.dataset_learn = IndexedSubset(self.dataset_learn.dataset, self.dataset_learn.indices)

        self.dataloader_labelled = DataLoader(self.dataset_labelled, batch_size=64, shuffle=True, drop_last=True)
        self.dataloader_learn = DataLoader(self.dataset_learn, batch_size=1, shuffle=False)

        if self.state_type == 'unsortedsim' or self.state_type == 'sortedsim' or 'sr1' in  self.state_type:
            labelled_indices = np.array(self.dataset_labelled.indices)
            labels = self.dataset.dataset.tensors[1][labelled_indices]
            _, meta_indices_sorted = torch.sort(labels)
            labelled_indices_sorted = labelled_indices[meta_indices_sorted]
            labelled_feature_sorted = self.dataset.dataset.tensors[0][labelled_indices_sorted]

            emb_class_avg = torch.zeros((10, 384))
            for i, idx in enumerate(range(0, len(labelled_feature_sorted), self.labelled_samples_per_class)):
                emb_class_avg[i] = labelled_feature_sorted[idx: idx+self.labelled_samples_per_class].sum(dim=0) / self.labelled_samples_per_class
            norm = torch.linalg.norm(emb_class_avg, ord=2, dim=1)
            emb_class_avg_norm = emb_class_avg / norm.unsqueeze(1)
            emb_class_avg_norm = emb_class_avg_norm.to(DEVICE)
            self.dataset_class_similarity = (self.dataset_complete_normed @ emb_class_avg_norm.T).cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_obs(self):
        # _image = self._image.cpu().numpy()[0]
        _pred = self._predictions[0]
        _pred_sorted = np.sort(self._predictions[0])
        _entropy = self._entropy
        _margin_score = [_pred_sorted[-1] - _pred_sorted[-2]]

        if self.state_type == 'sr2_womargin':
            return np.concatenate((_pred, _entropy), dtype=np.float32)
        elif self.state_type == 'sr1_womargin':
            _class_similarity = self.dataset_class_similarity[self._index[0]]
            return np.concatenate((_class_similarity, _entropy), dtype=np.float32)
        elif self.state_type == 'unsortedsim':
            _class_similarity = self.dataset_class_similarity[self._index[0]]
            return np.concatenate((_class_similarity, _entropy, _margin_score), dtype=np.float32)
        elif self.state_type == 'sortedsim':
            _class_similarity = self.dataset_class_similarity[self._index[0]]
            _class_similarity.sort()
            return np.concatenate((_class_similarity, _entropy, _margin_score), dtype=np.float32)
        elif self.state_type == 'prob_preds':
            return np.concatenate((_pred, _entropy, _margin_score), dtype=np.float32)

    def _get_info(self, action: Optional[int] = None):
        # # model_entropy[test_data_eps_avg] and model_entropy[test_data] is updated every time a label is queried
        info = {'model_performance':
                    {'learned_data': self.performance * 100,
                     'test_data': self._test_accuracy * 100},
                'model_entropy':
                    {
                        'current_obs': self._entropy[0],
                        'learn_data': self._entropy_reward,
                        'test_data': self._entropy_test,
                    },
                'queried_count': self.queried_count,
                'labelled_count': self.queried_count + self.init_labelled_count,
                }
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.queried_count = 0
        self.step_count = 0
        self.action_count = 0
        self._entropy_test = None
        self._entropy_test_eps = []
        # reset dataloaders
        self._create_dataloaders()
        self.train_iter = iter(self.dataloader_learn)
        self._image, self._label, self._index = next(self.train_iter)
        # self._image_normed = self.dataset_complete_normed[self._index[0]]
        self.performance = self.get_performance()

        # self._class_similarity = self._class_similarity / norm
        self._predictions, self._entropy = self.classifier_model.get_probabilities(self._image)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def get_performance(self):
        self.classifier_model.reset_model()
        self.classifier_model.train_model(self.dataloader_labelled, self.queried_count)
        # _, __ = self.classifier_model.test_score(self.dataloader_labelled)
        score, self._entropy_reward = self.classifier_model.test_score(self.dataloader_reward)
        self._test_accuracy, self._entropy_test = self.classifier_model.test_score(self.dataloader_test)
        self._entropy_test_eps.append(self._entropy_test)
        if self.verbose:
            print(f"queried data: {self.queried_count}, score: {score}")
        return score

    def query(self):
        self.dataset_labelled.indices.extend(self._index.tolist())
        self.dataloader_labelled = DataLoader(self.dataset_labelled, batch_size=64, shuffle=True, drop_last=True)
        self.queried_count += 1

    def step(self, action):
        if action == 1:
            self.query()
            new_performance = self.get_performance()
            reward_scale_factor = 100
            reward = (new_performance - self.performance) * reward_scale_factor
            if new_performance != self.performance:
                self.performance = new_performance
            self.action_count += 1
        else:
            reward = 0
            self.action_count -= 1

        if self.queried_count == self.al_budget:
            terminated = True
        else:
            terminated = False
            try:
                self._image, self._label, self._index = next(self.train_iter)
            except:
                terminated = True
                print('Dataloader for learning is out of samples. Setting terminated as True')
        self.step_count += 1
        # if self.eval_env and self.step_count == 500 and self.action_count == 500 or self.action_count == -500:
        #     print("First 500 steps have the same action. Ending eval episode.")
        #     terminated = True

        self._predictions, self._entropy = self.classifier_model.get_probabilities(self._image)
        observation = self._get_obs()
        info = self._get_info(action)

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass


def env(model, dataset, test_dataset, al_budget, episode_count, model_kwargs=None):
    gym_env = gym.make('custom_envs/ActiveImageClassifier-v1',
                       model=model,
                       dataset=dataset,
                       test_dataset=test_dataset,
                       al_budget=al_budget,
                       model_kwargs=model_kwargs)

    obs, info = gym_env.reset()

    eps_count = 0
    while eps_count < episode_count:
        action = gym_env.action_space.sample()
        next_obs, reward, terminated, _, info = gym_env.step(action)  # terminated is the same as done here

        if terminated:
            print(f"completed episode: {eps_count + 1}")
            obs, info = gym_env.reset()
            eps_count += 1
        # save obs, next_obs, rewards, terminated in a buffer
        obs = next_obs


def env_vectorised(model, dataloaders, al_budget, episode_count, env_count, model_kwargs):
    # set context for python multiprocessing
    vec_kwargs = {
        'context': 'spawn'
    }
    envs = gym.make_vec('custom_envs/ActiveImageClassifier-v1',
                        num_envs=env_count,
                        vector_kwargs=vec_kwargs,
                        model=model,
                        dataloaders=dataloaders,
                        al_budget=al_budget,
                        model_kwargs=model_kwargs)
    obs, infos = envs.reset(seed=42)
    eps_count = 0
    while eps_count < episode_count:
        actions = np.random.randint(2, size=(env_count,))
        next_obs, rewards, dones, _, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        if "_final_observation" in infos:
            for i, d in enumerate(dones):
                if d:
                    print(f"completed episode: {eps_count + 1}")
                    real_next_obs[i] = infos['final_observation'][i]
                    eps_count += 1
        # save obs, real_next_ob, rewards, terminated in a buffer
        obs = next_obs
    envs.close()  # to release resources


if __name__ == '__main__':
    from models.models import lenet5_og
    from models.wrapped_models import WrappedLenet5

    wrapped_CNN_model = WrappedLenet5(model=lenet5_og(),
                                      criterion=torch.nn.CrossEntropyLoss(),
                                      weight_decay=1e-8,
                                      learning_rate=1.5e-3,
                                      num_epochs=30
                                      )

    datasets = get_dataloaders('mnist')

    # dataset_train = MNIST(root=torch_data_path, train=True, download=True,
    #                       transform=transforms.Compose([transforms.ToTensor()]))
    # dataset_test = MNIST(root=torch_data_path, train=False, download=True,
    #                      transform=transforms.Compose([transforms.ToTensor()]))

    # train_dataloader = DataLoader(dataset_train, batch_size=1,
    #                               shuffle=True, num_workers=0, )
    # test_dataloader = DataLoader(dataset_test, batch_size=256,
    #                              shuffle=False, num_workers=0)
    #
    # dataloaders = (train_dataloader, test_dataloader)

    # model_kwargs = {
    #     'reduced_epoch_factor': 0.5,
    #     'epoch_queried_count_threshold': 5
    # }

    start_time = time.time()
    env(wrapped_CNN_model, dataset=datasets[0][1][0], test_dataset=datasets[0][1][2], al_budget=10, episode_count=10)
    finish_time_serial = time.time() - start_time

    # start_time = time.time()
    # env_vectorised(wrapped_CNN_model, dataloaders, al_budget=10, episode_count=10, env_count=4,
    #                model_kwargs=model_kwargs)
    # finish_time_vec = time.time() - start_time
    # print(f"Execution time: {finish_time_serial}")
    # print(f"Execution time: {finish_time_vec}")
