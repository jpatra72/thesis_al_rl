"""Dataloader of the small-scale continual learning experiment"""
import os
import copy
import random
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import KMNIST, MNIST, CIFAR10, SVHN

from tools.helper import DEVICE

from pbnn.src.pbnn.utils import seed
from tools.not_mnist import NotMNIST
from tools.all_mnist import AllMNIST
from tools.helper import torch_data_path


# TODO: is validation set required for AL?
def get_train_val_test_split_dataloaders(dataset_class: type,
                                         torch_data_path: str,
                                         split: List,
                                         transform,
                                         # create_initial_dataloader: bool = False,
                                         # class_cnt_initial_dataloader: int = 2,
                                         set_seed: bool = False,
                                         batch_sizes: Union[int, List[int]] = 1,
                                         num_workers: int = 0,
                                         pin_memory: bool = None) \
        -> Tuple[Dataset, Dataset, Dataset]:
    """Returns the training, validation and test split for a dataset.

    Args:
        dataset_class: A dataset class
        torch_data_path: The path to the torch datasets
        split: A list of [training split, validation split] for the split
            parameter of the dataset class
        transform: A torchvision transform
        batch_sizes: The batch sizes for the train, val and test dataloaders. If int, all batch_size is same for all
            dataloaders

    Returns:
        The train-, val- and test-dataloader
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    train_val = dataset_class(torch_data_path, split[0], transform=transform, download=True)
    train_size = int(1.0 * len(train_val))
    val_size = len(train_val) - train_size
    generator = torch.Generator().manual_seed(seed) if set_seed else None
    train, val = random_split(train_val, [train_size, val_size],
                              generator=generator)

    # train = IndexedSubset(train.dataset, train.indices)
    # if create_initial_dataloader:
    #     initial, train = get_balanced_datasets(train, class_cnt_initial_dataloader)
    #     initial_size = len(initial)
    #     train_size = train_size - initial_size
    # else:
    #     initial = Subset(train.dataset, [])
    #     initial_size = 0
    #     dataloader_initial = None
    # if isinstance(batch_sizes, list):
    #     batch_size_train, batch_size_val, batch_size_test = batch_sizes
    # else:
    #     batch_size_train, batch_size_val, batch_size_test = batch_sizes, batch_sizes, batch_sizes
    # if create_initial_dataloader:
    #     dataloader_initial = torch.utils.data.DataLoader(initial, pin_memory=pin_memory, num_workers=num_workers,
    #                                                  batch_size=batch_size_train, shuffle=True, generator=generator)
    # dataloader_train = torch.utils.data.DataLoader(train, pin_memory=pin_memory, num_workers=num_workers,
    #                                                batch_size=batch_size_train, shuffle=True, generator=generator)
    # dataloader_val = torch.utils.data.DataLoader(val, pin_memory=pin_memory, num_workers=num_workers,
    #                                              batch_size=batch_size_val, shuffle=False, generator=generator)
    test = dataset_class(torch_data_path, split[1], transform=transform, download=True)
    test_size = len(test)
    # dataloader_test = torch.utils.data.DataLoader(test, pin_memory=pin_memory, num_workers=num_workers,
    #                                               batch_size=batch_size_test, shuffle=False, generator=generator)
    print(f'{dataset_class.__name__}: Train: {train_size}, Val: {val_size}, Test: {test_size}')
    return train, val, test


def get_train_val_test_feature_split_dataloaders_resnet(dataset_name: str,
                                                          torch_data_path: str,
                                                          seed: int,
                                                          set_seed: bool = True,
                                                          train_val_split: int = 3,
                                                        resnet_type: int =50):
    train_val_pth_path = os.path.join(torch_data_path, f'resnet{resnet_type}_extracted_dataset',
                                      f'{dataset_name.lower()}_train_data.pt')
    test_pth_path = os.path.join(torch_data_path, f'resnet{resnet_type}_extracted_dataset', f'{dataset_name.lower()}_test_data.pt')

    train_val = torch.load(train_val_pth_path)
    train_size = int(train_val_split * len(train_val) / 100)
    val_size = len(train_val) - train_size
    generator = torch.Generator().manual_seed(seed) if set_seed else None
    train, val = random_split(train_val, [train_size, val_size],
                              generator=generator)
    test = torch.load(test_pth_path)
    test_size = len(test)

    print(f'{dataset_name.upper()}FeatureExtracted: Train: {train_size}, Val: {val_size}, Test: {test_size}')
    return train, val, test

def get_train_val_test_feature_split_dataloaders_dino(dataset_name: str,
                                                      torch_data_path: str,
                                                      # seed: int,
                                                      set_seed: bool = True,
                                                      train_val_split: int = 3, ):
    train_val_pth_path = os.path.join(torch_data_path, 'dino_extracted_dataset',
                                      f'{dataset_name.lower()}_train_data.pt')
    test_pth_path = os.path.join(torch_data_path, 'dino_extracted_dataset', f'{dataset_name.lower()}_test_data.pt')

    train_val = torch.load(train_val_pth_path)
    train_size = int(train_val_split * len(train_val) / 100)
    val_size = len(train_val) - train_size
    generator = torch.Generator().manual_seed(seed) if set_seed else None
    train, val = random_split(train_val, [train_size, val_size],
                              generator=generator)
    test = torch.load(test_pth_path)
    test_size = len(test)

    print(f'{dataset_name.upper()}FeatureExtracted: Train: {train_size}, Val: {val_size}, Test: {test_size}')
    return train, val, test


def get_dataloaders(
        dataset_order: Union[List[str], str],
        seed: int,
        batch_sizes: Union[int, List[int]] = None,
        feature_extracted: bool = False,
        train_val_split: int = 3,
        create_initial_dataloader: bool = False,
        class_cnt_initial_dataloader: int = 2,


) -> List[Tuple[Optional[int], Tuple[Dataset, Dataset, Dataset], torch.nn.Module]]:
    """Returns the dataloader of the small-scale continual learning experiment.

    Args:
        batch_sizes: The batch size

    Returns:
        A tuple of output dimensions and train-, val-, and test-dataloaders
    """

    def to_3_channel_grayscale(x):
        """Repeats the first channel to obtain a 3-channel grayscale image."""
        return x.repeat(3, 1, 1)

    dataset_name_to_class = {'mnist': MNIST,
                             'notmnist': NotMNIST,
                             'kmnist': KMNIST,
                             'allmnist': AllMNIST,
                             'cifar': CIFAR10,
                             'svhn': SVHN}

    if not isinstance(dataset_order, list):
        dataset_order = [dataset_order]

    datasets = [
        {'dataset_class': dataset_name_to_class[dataset_name.lower()],
         'split': [True, False],
         'transform': transforms.Compose([transforms.ToTensor(), ])
         } for dataset_name in dataset_order
    ]

    if feature_extracted:
        dataloaders = [
            (
                # len(dataset['dataset_class'].classes),
                10,
                get_train_val_test_feature_split_dataloaders_dino(dataset_name=dataset_order[idx].lower(),
                                                                  torch_data_path=torch_data_path,
                                                                  train_val_split=train_val_split,
                                                                  # seed=seed,
                                                                  ),
                torch.nn.CrossEntropyLoss())
            for idx, dataset in enumerate(datasets)]
    else:
        dataloaders = [(len(dataset['dataset_class'].classes),
                        get_train_val_test_split_dataloaders(torch_data_path=torch_data_path,
                                                             # batch_sizes=batch_sizes,
                                                             # create_initial_dataloader=create_initial_dataloader,
                                                             # class_cnt_initial_dataloader=class_cnt_initial_dataloader,
                                                             **dataset),
                        torch.nn.CrossEntropyLoss())
                       for dataset in datasets]
    return dataloaders


def get_feature_dataloaders(
        image_dataloaders: List[
            Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader, DataLoader], torch.nn.Module]],
        feature_extractor: torch.nn.Module,
        set_seed: bool = False,
        batch_sizes: Union[int, List[int]] = 1,
        num_workers: int = 0,
        pin_memory: bool = None
) -> List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader, DataLoader], torch.nn.Module]]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes] * 4

    generator = torch.Generator().manual_seed(seed) if set_seed else None
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    feature_dataloaders_list = []
    with torch.no_grad():
        for dataloader_tuple in image_dataloaders:
            new_dataloaders = tuple()
            dataset_sizes = []
            for idx, dataloader in enumerate(dataloader_tuple[1]):
                features = []
                targets = []
                if dataloader is not None:
                    for image, target in dataloader:
                        image = image.to(DEVICE)
                        feature = feature_extractor(image)
                        features.append(feature.cpu())
                        targets.append(target)
                    features_tensors = torch.cat(features, dim=0)
                    targets_tensors = torch.cat(targets, dim=0)
                    if isinstance(dataloader.dataset, torch.utils.data.Subset):
                        dataset_name = type(dataloader.dataset.dataset).__name__
                    else:
                        dataset_name = type(dataloader.dataset).__name__
                    dataset_name = f"feature{dataset_name}"

                    dataset = NamedTensorDataset(dataset_name, features_tensors, targets_tensors)
                    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, num_workers=num_workers,
                                                             batch_size=batch_sizes[idx], shuffle=True,
                                                             generator=generator)
                    dataset_sizes.append(len(dataset))
                else:
                    dataloader = dataloader
                    dataset_sizes.append(0)
                new_dataloaders += (dataloader,)

            print(
                f'Feature extracted {dataset.name}: Train: {dataset_sizes[0]}, Val: {dataset_sizes[1]}, Test: {dataset_sizes[2]}')
            new_dataloader_tuple = (dataloader_tuple[0], new_dataloaders, dataloader_tuple[2])
            feature_dataloaders_list.append(new_dataloader_tuple)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_dataloaders_list


def reset_dataloader_batch_sizes(
        dataloaders: List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], torch.nn.Module]],
        new_batch_sizes: List[int]
) -> List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], torch.nn.Module]]:
    dataloader_attr = ['dataset', 'num_workers', 'pin_memory', 'generator']
    new_dataloaders = []
    for dataloader_tuple in dataloaders:
        new_dataloaders_tuple = tuple()
        for idx, dataloader in enumerate(dataloader_tuple[1]):
            dataloader_kwargs = {attr: getattr(dataloader, attr) for attr in dataloader_attr}
            new_dataloader = torch.utils.data.DataLoader(batch_size=new_batch_sizes[idx],
                                                         shuffle=True,
                                                         **dataloader_kwargs)
            new_dataloaders_tuple += (new_dataloader,)
        new_dataloader_tuple = (dataloader_tuple[0], new_dataloaders_tuple, dataloader_tuple[2])
        new_dataloaders.append(new_dataloader_tuple)
    return new_dataloaders


class NamedTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, name, *args):
        super().__init__(*args)
        self.name = name


def get_dataloader_name(
        dataloader: Union[
            Tuple[Optional[int], Tuple[Optional[DataLoader], DataLoader, DataLoader, DataLoader], torch.nn.Module],
            DataLoader]
) -> str:
    if isinstance(dataloader, Tuple):
        dataloader = dataloader[1][1]
    if isinstance(dataloader.dataset, NamedTensorDataset):
        return dataloader.dataset.name
    elif isinstance(dataloader.dataset, torch.utils.data.Subset):
        return type(dataloader.dataset.dataset).__name__
    else:
        return type(dataloader.dataset).__name__


def get_balanced_datasets(dataset: Subset, samples_per_class: int = 2, shuffle: bool = False) \
        -> (Subset, Subset):
    original_indices = copy.deepcopy(dataset.indices)
    if shuffle:
        np.random.shuffle(original_indices)
    underlying_dataset = dataset.dataset
    if hasattr(underlying_dataset, 'classes'):
        num_classes = len(underlying_dataset.classes)
    elif hasattr(underlying_dataset, 'tensors'):
        num_classes = len(torch.unique(underlying_dataset.tensors[1]))

    selected_indices = []
    samples_needed_per_class = [samples_per_class] * num_classes

    for index in list(original_indices):
        if hasattr(underlying_dataset, 'targets'):
            class_label = underlying_dataset.targets[index]
        elif hasattr(underlying_dataset, 'tensors'):
            class_label = underlying_dataset.tensors[1][index]

        if samples_needed_per_class[class_label] != 0 and sum(samples_needed_per_class) != 0:
            selected_indices.append(index)
            original_indices.remove(index)
            samples_needed_per_class[class_label] -= 1
        elif sum(samples_needed_per_class) == 0:
            break

    selected_dataset = Subset(underlying_dataset, selected_indices)
    remaining_dataset = Subset(underlying_dataset, original_indices)

    return selected_dataset, remaining_dataset


class IndexedSubset(Subset):
    def __init__(self, dataset: Dataset, indices: List):
        super().__init__(dataset=dataset, indices=indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]], self.indices[idx]

    def __getitems__(self, indices: List[int]):
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] + (self.indices[idx],) for idx in indices]
