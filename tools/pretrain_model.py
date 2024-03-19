import os
import time
from typing import Callable, Any, Union, List, Optional

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from tools.utils import fit
from helper import model_folder_path, logs_path
from mnist_dataloaders import get_dataloaders, get_dataloader_name


def train_model(dataset_name: str,
                model_function: Callable[[Any], nn.Module],
                model_name_prefix: str,
                batch_sizes: Union[List[int], int],
                retrain: bool = False,
                weight_decay: float = 0.0,
                learning_rate: float = 2e-3,
                num_epochs: int = 100,
                patience: int = 10,
                metrics_to_log: Union[List[str], str] = None,
                *model_function_args,
                **model_function_kwargs
                ) -> (nn.Module, Any):
    dataloader = get_dataloaders(dataset_name, batch_sizes)
    dataset_name = get_dataloader_name(dataloader[0])

    model_name = f'{model_name_prefix}_{dataset_name.lower()}'
    model_path = os.path.join(model_folder_path, f'{model_name}.pth')

    if os.path.exists(model_path) and not retrain:
        return model_function(*model_function_args, **model_function_kwargs), None

    if retrain:
        model_function_kwargs['pretrained'] = False
        model_function_kwargs['pth_file_name'] = None
        model = model_function(*model_function_args,
                               **model_function_kwargs)
        loss_function = dataloader[0][2]
        model.requires_grad_(True)
        metrics = fit(model, dataloader[0][1][1:], loss_function, weight_decay,
                      is_classification=type(loss_function) is nn.CrossEntropyLoss,
                      learning_rate=learning_rate, num_epochs=num_epochs, patience=patience,
                      metrics_run_epoch=metrics_to_log)

        torch.save(model.state_dict(), model_path)

        return model, metrics


if __name__ == '__main__':
    from models.models import lenet5, lenet5_og, lenet5_30

    result_path = 'lenet_mnist_results'
    metrics_over_runs = {}

    for weight_decay in [1e-8]:
        print(f"Current run weight_decay: {weight_decay}")
        model, metrics = train_model(dataset_name='mnist',
                                     model_function=lenet5_og,
                                     model_name_prefix='lenet5_og',
                                     batch_sizes=[256, 256, 256],
                                     retrain=True,
                                     num_epochs=100,
                                     weight_decay=weight_decay,
                                     learning_rate=2e-3,
                                     patience=20
                                     # metrics_to_log=['accuracy', 'entropy']
                                     )
        metrics_over_runs[str(weight_decay)] = metrics

    if result_path is not None and metrics is not None:
        result_path = os.path.join(logs_path, result_path, f'metric_results_{time.time()}')
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        with open(result_path, 'w') as f:
            json.dump(metrics_over_runs, f, indent=4, default=str)
