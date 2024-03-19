import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.helper import base_path, DEVICE


def model_path(model_name: str):
    return os.path.join(base_path, 'models', model_name)


class Lenet5(nn.Module):
    def __init__(self, pretrained=False, pth_file_name=None):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )
        if pretrained:
            state_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{pth_file_name}.pth")
            state_dict = torch.load(state_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


def lenet5_og(pretrained=False, pth_file_name: str = None):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Dropout(0.25),
        torch.nn.Conv2d(16, 120, 5),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(84, 10)
    )

    if pretrained:
        state_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"lenet5_{pth_file_name.lower()}.pth")
        state_dict = torch.load(state_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    return model


def lenet5(pretrained=False, pth_file_name: str = None):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10)
    )

    if pretrained:
        state_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"lenet5_{pth_file_name.lower()}.pth")
        state_dict = torch.load(state_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    return model


def lenet5_30(pretrained=False, pth_file_name: str = None):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 30)
    )

    if pretrained:
        state_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"lenet5_30_{pth_file_name.lower()}.pth")
        state_dict = torch.load(state_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    return model


def fcnn(in_feature: int):
    model = nn.Sequential(
        torch.nn.Linear(in_feature, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )

    return model


def fcnn_1layer(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(384, 10),
    )
    return model

def fcnn_2layer(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(384, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    return model

def fcnn_4layer(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(384, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    return model


def fcnn_1layer_resnet50(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(2048, 10),
    )
    return model

def fcnn_2layer_resnet50(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(2048, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    return model


def fcnn_2layer_resnet34(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(512, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    return model


def fcnn_3layer(pretrained=False, pth_file_name: str = None):
    model = nn.Sequential(
        torch.nn.Linear(384, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    return model


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


model_name_to_function_dict = {'lenet5': lenet5,
                               'lenet5_30': lenet5_30,
                               'lenet5_og': lenet5_og,
                               'fcnn_1layer': fcnn_1layer,
                               'fcnn_2layer': fcnn_2layer,
                               'fcnn_3layer': fcnn_3layer,
                               'fcnn_4layer': fcnn_4layer,
                                'fcnn_1layer_resnet50': fcnn_1layer_resnet50,
                               'fcnn_2layer_resnet50': fcnn_2layer_resnet50,
                               'fcnn_2layer_resnet34': fcnn_2layer_resnet34}
