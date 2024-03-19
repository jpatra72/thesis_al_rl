import os
import argparse
import time

import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tools.mnist_dataloaders import get_train_val_test_feature_split_dataloaders_resnet, \
    get_train_val_test_feature_split_dataloaders_dino
from tools.helper import torch_data_path, get_logs_subfolder_path, wandb_folder_path, set_random_seed
from models.wrapped_models import WrappedLenet5
from models.models import model_name_to_function_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--wandb_logging', type=bool, default=False)
parser.add_argument('--model', type=str, default='dino_single')
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()


def main(dataset, seed, train_val_split, model):
    if model == 'resnet50':
        network = model_name_to_function_dict['fcnn_1layer_resnet50']()
        train, val, test = get_train_val_test_feature_split_dataloaders_resnet(dataset_name=dataset,
                                                                               torch_data_path=torch_data_path,
                                                                               seed=seed,
                                                                               set_seed=True,
                                                                               train_val_split=train_val_split,
                                                                               resnet_type=50)
    elif model == 'resnet34':
        network = model_name_to_function_dict['fcnn_2layer_resnet34']()
        train, val, test = get_train_val_test_feature_split_dataloaders_resnet(dataset_name=dataset,
                                                                               torch_data_path=torch_data_path,
                                                                               seed=seed,
                                                                               set_seed=True,
                                                                               train_val_split=train_val_split,
                                                                               resnet_type=34)
    elif model == 'dino_single':
        network = model_name_to_function_dict['fcnn_1layer']()
        train, val, test = get_train_val_test_feature_split_dataloaders_dino(dataset_name=dataset,
                                                                             torch_data_path=torch_data_path,
                                                                             # seed=seed,
                                                                             set_seed=True,
                                                                             train_val_split=train_val_split)

    elif model == 'dino_multi':
        network = model_name_to_function_dict['fcnn_2layer']()
        train, val, test = get_train_val_test_feature_split_dataloaders_dino(dataset_name=dataset,
                                                                             torch_data_path=torch_data_path,
                                                                             # seed=seed,
                                                                             set_seed=True,
                                                                             train_val_split=train_val_split)

    # train_loader = DataLoader(train, batch_size=64, shuffle=True, pin_memory=True, num_workers=2)
    # test_loader = DataLoader(test, batch_size=512, shuffle=False, pin_memory=True, num_workers=2)

    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=512, shuffle=False)
    network = WrappedLenet5(model=network,
                            criterion=torch.nn.CrossEntropyLoss(),
                            weight_decay=1e-8,
                            learning_rate=1.5e-3,
                            num_epochs=50,
                            tqdm_bar_enable=True
                            )
    train_start = time.time()
    network.train_model(train_loader)

    train_time = time.time() - train_start

    test_acc, _ = network.test_score(test_loader)

    return test_acc, train_time


if __name__ == '__main__':
    set_random_seed(args.seed)
    run_name = f"{args.dataset}_{args.model}"
    if args.wandb_logging:
        run_name = f"{args.dataset}_{args.model}_1layer"
        wandb.init(project='dino_vs_resnet02', name=run_name, config=vars(args), sync_tensorboard=True,
                   dir=wandb_folder_path)
        print(f'wandb run name : {run_name}')
    tb_log_path = get_logs_subfolder_path(os.path.join('dino_vs_resnet', f'{run_name}_{args.seed}'))
    tb_writer = SummaryWriter(log_dir=tb_log_path)

    # for dataset in ['MNIST', 'KMNIST', 'CIFAR']:
    #     for seed in [1124, 343423, 987, 6789]:
    for idx, train_percent in enumerate([1, 10, 30, 50, 70, 90, 100]):
        test_accuracy, train_time = main(args.dataset, args.seed, train_percent, args.model)
        tb_writer.add_scalar(f'performance/test_accuracy', test_accuracy * 100,
                             global_step=idx)
        tb_writer.add_scalar(f'performance/training_time', train_time,
                             global_step=idx)
        tb_writer.add_scalar(f'step/data_percent', train_percent,
                             global_step=idx)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
