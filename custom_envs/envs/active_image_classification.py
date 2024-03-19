import time
from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from models.wrapped_models import BaseNetworkWrapper
from tools.helper import QueriedDataset
from tools.helper import torch_data_path


# TODO: use pin_memory?
class ImageClassifier(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 model: BaseNetworkWrapper,
                 dataloaders: Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], torch.nn.Module],
                 al_budget: int,
                 model_kwargs: Optional[Dict] = None,
                 verbose: bool = False):
        self.network_class_probs = model
        if model_kwargs:
            self.network_class_probs.set_values_from_env(**model_kwargs)
        self.al_budget = al_budget
        self.train_image_dataloader = dataloaders[1][0]
        self.test_image_dataloader = dataloaders[1][1]
        self.verbose = verbose

        assert self.train_image_dataloader.batch_size == 1, f"train_dataloader batch_size should be 1"
        assert self.test_image_dataloader.batch_size >= 256, (
            f"set test_dataloader batch_size > 256 for efficient GPU use."
            f" Current test_dataloader batch_size is {self.test_image_dataloader.batch_size}")

        self.queried_count = 0
        self.performance = 0.0
        self._entropy_test = None
        self._entropy_test_eps = []
        self._entropy_test_eps_avg = 0
        self.create_queried_dataloader_flag = True

        self.image_shape = next(iter(self.train_image_dataloader))[0].shape  # shape = (1, n_channels, n_height, n_width)
        self.class_count = dataloaders[0]

        # Observations are dictionaries containing the image, model's class predictions and confidence.
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 1, shape=self.image_shape[1:], dtype=np.float32),
                "predictions": spaces.Box(0, 1, shape=(self.class_count,), dtype=np.float32),
                "confidence": spaces.Box(float('-inf'), float('inf'), shape=(1,), dtype=np.float32)
            }
        )

        # We have 2 actions, corresponding to 0, 1 corresponding to do not query label and query label respectively
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        return {"image": self._image.cpu().numpy()[0], "predictions": self._predictions[0], "confidence": self._confidence}

    def _get_info(self, action: Optional[int] = None):
        # model_entropy[test_data_eps_avg] and model_entropy[test_data] is updated every time a label is queried
        if bool(action):
            self._entropy_test_eps_avg = sum(self._entropy_test_eps[-20:]) / len(self._entropy_test_eps[-20:])
        info = {'model_performance': self.performance * 100,
                'model_entropy':
                    {
                        'current_obs': self._confidence[0],
                        # following entropies are updated every time a new label is queried,
                        'test_data': self._entropy_test,
                    'test_data_eps_avg': self._entropy_test_eps_avg,
                    },
                'queried_count': self.queried_count
                }
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.queried_count = 0
        self.performance = 0.0
        self.create_queried_dataloader_flag = True
        # reset train dataloader
        self.train_iter = iter(self.train_image_dataloader)
        self._image, self._label = next(self.train_iter)
        self.network_class_probs.reset_model()
        self._predictions, self._confidence = self.network_class_probs.get_probabilities(self._image)

        observation = self._get_obs()
        info = self._get_info()

        self._entropy_test = None
        self._entropy_test_eps = []
        self._entropy_test_eps_avg = None
        return observation, info

    def get_performance(self):
        self.network_class_probs.reset_model()
        self.network_class_probs.train_model(self.queried_dataloader, self.queried_count)
        score, self._entropy_test = self.network_class_probs.test_score(self.test_image_dataloader)
        self._entropy_test_eps.append(self._entropy_test)
        if self.verbose:
            print(f"queried data: {self.queried_dataloader.dataset.queried_count}, score: {score}")
        return score

    def query(self):
        # query label and add data to the queried_dataset
        if self.create_queried_dataloader_flag:
            self.create_queried_dataloader_flag = False
            self.queried_dataset = QueriedDataset(self._image, self._label, self.al_budget)
            self.queried_dataloader = DataLoader(self.queried_dataset, batch_size=512, shuffle=True)
        else:
            self.queried_dataset.append_new_data(self._image, self._label)
        self.queried_count += 1

    def step(self, action):
        if action == 1:
            self.query()
            new_performance = self.get_performance()
            reward_scale_factor = 100
            reward = (new_performance - self.performance) * reward_scale_factor
            if new_performance != self.performance:
                self.performance = new_performance
        else:
            reward = 0

        if self.queried_count == self.al_budget:
            terminated = True
        else:
            terminated = False
            try:
                self._image, self._label = next(self.train_iter)
            except StopIteration:
                # TODO: handle episode truncating
                raise Exception("Well, the gym env is out of samples to generate")

        self._predictions, self._confidence = self.network_class_probs.get_probabilities(self._image)
        observation = self._get_obs()
        info = self._get_info(action)
        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass


def single_env(model, dataloaders, al_budget, episode_count, model_kwargs):
    gym_env = gym.make('custom_envs/ActiveImageClassifier-v0',
                       model=model,
                       dataloaders=dataloaders,
                       al_budget=al_budget,
                       model_kwargs=model_kwargs)

    obs, info = gym_env.reset()

    eps_count = 0
    while eps_count < episode_count:
        action = gym_env.action_space.sample()
        next_obs, reward, terminated, _, info = gym_env.step(action)    # terminated is the same as done here

        if terminated:
            print(f"completed episode: {eps_count+1}")
            obs, info = gym_env.reset()
            eps_count += 1
        # save obs, next_obs, rewards, terminated in a buffer
        obs = next_obs


def env_vectorised(model, dataloaders, al_budget, episode_count, env_count, model_kwargs):
    # set context for python multiprocessing
    vec_kwargs = {
        'context': 'spawn'
    }
    envs = gym.make_vec('custom_envs/ActiveImageClassifier-v0',
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
                    print(f"completed episode: {eps_count+1}")
                    real_next_obs[i] = infos['final_observation'][i]
                    eps_count += 1
        # save obs, real_next_ob, rewards, terminated in a buffer
        obs = next_obs
    envs.close()    # to release resources


if __name__ == '__main__':

    from models.models import lenet5
    from models.wrapped_models import WrappedLenet5

    wrapped_CNN_model = WrappedLenet5(model=lenet5(),
                                      criterion=torch.nn.CrossEntropyLoss())

    dataset_train = MNIST(root=torch_data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    dataset_test = MNIST(root=torch_data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader = DataLoader(dataset_train, batch_size=1,
                                  shuffle=True, num_workers=0,)
    test_dataloader = DataLoader(dataset_test, batch_size=256,
                                 shuffle=False, num_workers=0)

    dataloaders = (train_dataloader, test_dataloader)

    model_kwargs = {
        'reduced_epoch_factor': 0.5,
        'epoch_queried_count_threshold': 5
    }

    start_time = time.time()
    single_env(wrapped_CNN_model, dataloaders, al_budget=10, episode_count=1, model_kwargs=model_kwargs)
    finish_time_serial = time.time() - start_time

    start_time = time.time()
    # env_vectorised(wrapped_CNN_model, dataloaders, al_budget=10, episode_count=1, env_count=4, model_kwargs=model_kwargs)
    finish_time_vec = time.time() - start_time
    print(f"Execution time: {finish_time_serial}")
    print(f"Execution time: {finish_time_vec}")
