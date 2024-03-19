import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from torchvision.datasets import KMNIST, MNIST, CIFAR10, SVHN
from tools.not_mnist import NotMNIST
from tools.helper import torch_data_path, DEVICE, model_folder_path


# class ReshapeTensor(torchvision.transforms.Trans):
#     def __call__(self, tensor):
#         return tensor.permute(0, 2, 3, 1)



transform_dict = {
    'mnist': transforms.Compose([
        transforms.Resize((210, 210)),
        transforms.ToTensor(),  # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.308,)),
    ]),
    'notmnist': transforms.Compose([
        transforms.Resize((210, 210)),
        transforms.ToTensor(),  # first, convert image to PyTorch tensor
        transforms.Normalize((0.4177,), (0.454,)),
    ]),
    'kmnist': transforms.Compose([
        transforms.Resize((210, 210)),
        transforms.ToTensor(),  # first, convert image to PyTorch tensor
        transforms.Normalize((0.1918,), (0.348,)),
    ]),
    'cifar': transforms.Compose([
        transforms.Resize((210, 210)),
        transforms.ToTensor(),  # first, convert image to PyTorch tensor
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ]),
    'svhn': transforms.Compose([
        transforms.Resize((210, 210)),
        transforms.ToTensor(),  # first, convert image to PyTorch tensor
        # ReshapeTensor(),
        transforms.Normalize([0.4377, 0.4438, 0.4728], [0.198, 0.201, 0.197]),
    ]),
}

dataset_dict = {
    'mnist': MNIST,
    'notmnist': NotMNIST,
    'kmnist': KMNIST,
    'cifar': CIFAR10,
    'svhn': SVHN,
}



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='svhn')
args = parser.parse_args()


def main():
    device = DEVICE
    print(f"device used: {device}")
    torch.hub.set_dir(model_folder_path)
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2_vits14.to(device)

    dataset = dataset_dict[args.dataset.lower()]
    transform = transform_dict[args.dataset.lower()]

    if args.dataset.lower() == 'svhn':
        train_dataset = dataset(root=torch_data_path, split='train', transform=transform, download=True)
        test_dataset = dataset(root=torch_data_path, split='test', transform=transform, download=True)
    else:
        train_dataset = dataset(root=torch_data_path, train=True, transform=transform, download=True)
        test_dataset = dataset(root=torch_data_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=2)

    dinov2_vits14.eval()
    with torch.no_grad():
        data = []
        targets = []
        for idx, loader in enumerate([train_loader, test_loader]):
            features = []
            labels = []
            count = 0
            for images, l in tqdm(loader, desc=f'Iteration: {idx}'):
                images = images.to(device)
                if 'mnist' in args.dataset.lower():
                    output = dinov2_vits14(images[:, :1].repeat(1, 3, 1, 1))
                else:
                    output = dinov2_vits14(images)

                output = output.cpu()
                features.append(output)
                labels.append(l)
                torch.cuda.empty_cache()
                count += 1
                # if count ==2:
                #     break
            data.append(torch.cat(features, dim=0).cpu())
            targets.append(torch.cat(labels, dim=0))

    train_dataset = torch.utils.data.TensorDataset(data[0], targets[0])
    test_dataset = torch.utils.data.TensorDataset(data[1], targets[1])
    save_path = os.path.join(torch_data_path, 'feature_extracted')
    torch.save(train_dataset, os.path.join(save_path, f'{args.dataset.lower()}_train_data.pt'))
    torch.save(test_dataset, os.path.join(save_path, f'{args.dataset.lower()}_test_data.pt'))


if __name__ == '__main__':
    main()