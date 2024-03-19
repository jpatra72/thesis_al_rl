import os
import sys
import argparse

os_type = sys.platform

import numpy as np
try:
    import tsnecuda as tsne
    print('using tsnecuda')
except:
    from sklearn.manifold import TSNE as tsne

    print('using sklearn')


# import matplotlib.pyplot as plt

from tools.mnist_dataloaders import get_train_val_test_feature_split_dataloaders_resnet, \
    get_train_val_test_feature_split_dataloaders_dino
from tools.helper import torch_data_path, get_logs_subfolder_path, wandb_folder_path, set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--model', type=str, default='resnet50')
args = parser.parse_args()


def main(dataset, train_val_split, model):
    if model == 'resnet50':
        _, _, test = get_train_val_test_feature_split_dataloaders_resnet(dataset_name=dataset,
                                                                           torch_data_path=torch_data_path,
                                                                           seed=152,
                                                                           set_seed=True,
                                                                           train_val_split=train_val_split,
                                                                           resnet_type=50)
    elif model == 'resnet34':
        _, _, test = get_train_val_test_feature_split_dataloaders_resnet(dataset_name=dataset,
                                                                           torch_data_path=torch_data_path,
                                                                           seed=152,
                                                                           set_seed=True,
                                                                           train_val_split=train_val_split,
                                                                           resnet_type=34)
    elif model == 'dino':
        _, _, test = get_train_val_test_feature_split_dataloaders_dino(dataset_name=dataset,
                                                                             torch_data_path=torch_data_path,
                                                                             seed=152,
                                                                             set_seed=True,
                                                                             train_val_split=train_val_split)

    x_test = test.tensors[0].numpy()
    y_test = test.tensors[1].numpy()

    tsne_result = tsne(perplexity=1000).fit_transform(x_test)
    # tsne_result = np.array([1, 2])

    save_path = get_logs_subfolder_path(os.path.join('tsne_exp', 'npdata'))
    file_path = os.path.join(save_path, f'{model}_{dataset}')
    np.savez(file_path, tsne_result=tsne_result, y_test=y_test)


    # plt.figure(figsize=(20, 15))
    # color_map = plt.cm.get_cmap('tab10')
    #
    # # plot without labels (faster)
    # plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_test, cmap=color_map)
    #
    # # plot labels
    # labels = np.array(np.arange(10))[y_test]
    # class_num = set()
    # for x1, x2, c, l in zip(tsne_result[:0], tsne_result[:1], color_map(y_test), labels):
    #     if len(class_num) == 10:
    #         break
    #     plt.scatter(x1, x2, c=[c], label=l)
    #     class_num.add(l)
    #
    # # remvoe duplicate labels
    # hand, labl = plt.gca().get_legend_handles_labels()
    # handout = []
    # lablout = []
    # for h, l in zip(hand, labl):
    #     if l not in lablout:
    #         lablout.append(l)
    #         handout.append(h)
    # plt.title(f'{model.title()}, {dataset.upper()}')
    # plt.xlabel('Component One')
    # plt.ylabel('Component Two')
    # plt.legend(handout, lablout, fontsize=20)
    # # plt.savefig(IMAGE_PATH + save_as)
    # plt.show()



if __name__ == '__main__':
    main(dataset=args.dataset, model=args.model, train_val_split=0)