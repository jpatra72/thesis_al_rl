import os
import glob
import yaml
import shutil
import random
import argparse

from typing import Optional, Tuple, List, Union, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

base_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
torch_data_path = os.path.join(base_path, 'data', 'torch')
logs_path = os.path.join(base_path, 'logs')
plots_path = os.path.join(base_path, 'plots')
model_folder_path = os.path.join(base_path, 'models')
wandb_folder_path = os.path.join(base_path, 'wandb')


def get_logs_subfolder_path(folder_name: str):
    subfolder_path = os.path.join(logs_path, folder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    return subfolder_path

def get_plots_subfolder_path(folder_name: str):
    subfolder_path = os.path.join(plots_path, folder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    return subfolder_path

def rename_folder(parent_dir: str, old_name: str, new_name: str)\
        -> (str, str):
    old_path = os.path.join(parent_dir, old_name)
    new_path = os.path.join(parent_dir, new_name)
    try:
        os.rename(old_path, new_path)
        print(f'Folder renamed from {old_path} to {new_path}')
    except Exception as e:
        print(f'Error: {e}')
    return new_path, new_name

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class QueriedDataset(Dataset):
    def __init__(self, image, label, al_budget):
        # TODO: pre-allocate image and label vars with AL budget size
        self.al_budget = al_budget
        self.images = np.empty(shape=(al_budget,) + image.shape[1:], dtype=np.float32)
        self.labels = np.empty(shape=(al_budget,), dtype=np.int64)
        self.images[0] = image
        self.labels[0] = label
        self.queried_count = 1

    def __len__(self):
        return self.queried_count

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def append_new_data(self, new_image, new_label):
        # add any new queried data to the training dataset
        self.images[self.queried_count] = new_image
        self.labels[self.queried_count] = new_label
        self.queried_count += 1


def split_network(network: nn.Module,
                  split_index: Optional[str] = None) \
        -> (nn.Module, nn.Module):
    # this method has to be generalised to other off-the-shelf network types (DINO model)

    if split_index is None:
        split_index = []
        for idx, layer in enumerate(network.children()):
            if isinstance(layer, nn.Flatten):
                split_index.append(idx)

    if split_index == []:
        return None, None

    if isinstance(network, nn.Sequential):
        layer_list = list(network)
        feature_network = nn.Sequential(*layer_list[:split_index[-1] + 1])
        head_network = nn.Sequential(*layer_list[split_index[-1] + 1:])
        return feature_network, head_network


def get_feature_vec_size(network: nn.Module,
                         feature_layer_index: Optional[str] = None) \
        -> int:
    if feature_layer_index is None and isinstance(network, nn.Sequential):
        feature_layer_index = []
        for idx, layer in enumerate(network.children()):
            if isinstance(layer, nn.Flatten):
                feature_layer_index.append(idx)
        layer_list = list(network)
        feature_vec_size = layer_list[feature_layer_index[-1] + 1].in_features
        return feature_vec_size


def get_run_log_path(logs_subfolder_path: str, run_name_prefix: str) -> (str, str):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param logs_subfolder_path: Path to the log folder containing several runs.
    :param run_name_prefix:
    :return:
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(logs_subfolder_path, f"{glob.escape(run_name_prefix)}_[0-9]*")):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if run_name_prefix == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    run_name = f'{run_name_prefix}_{max_run_id+1}'
    run_log_path = os.path.join(logs_subfolder_path, run_name)
    return run_log_path, run_name


def create_parser_from_yaml(path: str, default: Optional[str] = None):
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--yaml_file', type=str, default=default, help="the name of the yml file with hyperparams")
    args = parser.parse_args()
    args.yaml_file += '.yml'
    yaml_full_path = os.path.join(path, args.yaml_file)
    with open(yaml_full_path, 'r') as f:
        args_data = yaml.safe_load(f)

    for key, value in args_data.items():
        flags = [f'--{key}']
        arg_type = type(value)
        parser.add_argument(*flags, type=arg_type, default=value)
    args = parser.parse_args()
    return args


def copy_files_by_extension(source_folder, destination_folder, run_id, extension):
    source_log_path = os.path.join(logs_path, source_folder, f"DQN_{run_id}")
    destination_log_path = os.path.join(logs_path, destination_folder, f"DQN_{run_id}")
    os.makedirs(destination_log_path, exist_ok=True)
    files = os.listdir(source_log_path)

    for file in files:
        if file.endswith(extension):
            source_file = os.path.join(source_log_path, file)
            destination_file = os.path.join(destination_log_path, file)

            if os.path.exists(destination_file):
                print(f'File already exists: {destination_file}')
                continue

            # Copy the file to the destination folder
            shutil.copy2(source_file, destination_file)
            print(f'Copied: {file}')
