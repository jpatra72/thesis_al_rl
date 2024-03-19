"""Implementation of the NotMNIST Dataset."""
from typing import Optional, Callable, Tuple, Any

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets import MNIST, KMNIST

from tools.not_mnist import NotMNIST


class AllMNIST(VisionDataset):
    classes = [None] * 30

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False
    ) -> None:
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.train = train

        dataset_mnist = MNIST(root, self.train, download=download)
        dataset_notmnist = NotMNIST(root, self.train, download=download)
        dataset_kmnist = KMNIST(root, self.train, download=download)

        self.classes = dataset_mnist.classes + dataset_notmnist.classes + dataset_kmnist.classes
        self.data = torch.cat((dataset_mnist.data,
                               dataset_notmnist.data,
                               dataset_kmnist.data),
                              dim=0)
        self.targets = torch.cat((dataset_mnist.targets,
                                  dataset_notmnist.targets + 10,
                                  dataset_kmnist.targets + 20),
                                 dim=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
