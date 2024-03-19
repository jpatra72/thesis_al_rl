import os
import numpy as np

from torchvision import datasets, transforms
from tools.not_mnist import NotMNIST

from tools.helper import torch_data_path

train_transform = transforms.Compose([transforms.ToTensor()])

# cifar10
train_set = datasets.CIFAR10(root=torch_data_path, train=True, download=True, transform=train_transform)
print(train_set.__class__)
print(train_set.data.shape)
print(train_set.data.mean(axis=(0, 1, 2)) / 255)
print(train_set.data.std(axis=(0, 1, 2)) / 255)
# (50000, 32, 32, 3)
# [0.49139968  0.48215841  0.44653091]
# [0.24703223  0.24348513  0.26158784]

# SVHN
train_set = datasets.SVHN(root=torch_data_path, split='train', download=True, transform=train_transform)
print(train_set.__class__)
print(list(train_set.data.shape))
print(train_set.data.mean(axis=(0, 2, 3)) / 255)
print(train_set.data.std(axis=(0, 2, 3)) / 255)
# [73257, 3, 32, 32]
# [0.4376821 , 0.4437697 , 0.47280442]
# [0.19803012, 0.20101562, 0.19703614]


# mnist
train_set = datasets.MNIST(root=torch_data_path, train=True, download=True, transform=train_transform)
print(train_set.__class__)
print(list(train_set.data.size()))
print(train_set.data.float().mean() / 255)
print(train_set.data.float().std() / 255)
# [60000, 28, 28]
# 0.1307
# 0.3081

# kmnist
train_set = datasets.KMNIST(root=torch_data_path, train=True, download=True, transform=train_transform)
print(list(train_set.data.size()))
print(train_set.data.float().mean() / 255)
print(train_set.data.float().std() / 255)
# [60000, 28, 28]
# 0.1918
# 0.3483


# notmnist
train_set = NotMNIST(root=torch_data_path, train=True, download=True, transform=train_transform)
print(train_set.__class__)
print(list(train_set.data.size()))
print(train_set.data.float().mean() / 255)
print(train_set.data.float().std() / 255)
# [60000, 28, 28]
# 0.4177
# 0.4540
