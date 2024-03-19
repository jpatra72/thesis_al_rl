import argparse
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from viz_utils.calculate_stuff import calculate_gp_posterior
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors

from tools.helper import plots_path, get_plots_subfolder_path, logs_path


performance_log_path = os.path.join(logs_path, 'dino_vs_resnet', 'wandb_logs')
datasets = ['cifar10', 'mnist', 'kmnist']
labels = {
    'dino_multi': 'DINOv2 Multi Layer',
    'dino_single':'DINOv2 Single Layer',
    'resnet50': 'ResNet50 Single Layer',
    'resnet34': 'ResNet34 Single Layer',
}
c = load_colors()
c4_100_keys = [key for key in c if '100' in key if key[:-3] in ['tuerkis', 'gruen', 'orange',
                                                                # 'petrol'
                                                                ]]
c4_75_keys = [key for key in c if '75' in key if key[:-2] in ['tuerkis', 'gruen', 'orange',
                                                              # 'petrol'
                                                              ]]
c4_25_keys = [key for key in c if '25' in key if key[:-2] in ['tuerkis', 'gruen', 'orange',
                                                              # 'petrol'
                                                              ]]

def plot_performance(models, dataset, df, ax):
    data_percent = np.array([1, 10, 30, 50, 70, 90, 100]) / 100
    performance_mean = []
    performance_min = []
    performance_max = []


    for model in models:
        model_columns = [f'Name: {dataset}_{model} - performance/test_accuracy',
                         f'Name: {dataset}_{model} - performance/test_accuracy__MIN',
                         f'Name: {dataset}_{model} - performance/test_accuracy__MAX'
                         ]
        performance_mean.append(df[model_columns[0]].tolist())
        performance_min.append(df[model_columns[1]].tolist())
        performance_max.append(df[model_columns[2]].tolist())



    for idx in range(len(performance_mean)):
        ax.plot(data_percent, performance_mean[idx], color=c[c4_100_keys[idx]], label=labels[models[idx]],
                 marker='o', markersize=2.5, linewidth=1)
        ax.fill_between(data_percent, performance_min[idx], performance_max[idx],
                         color=c[c4_25_keys[idx]], alpha=0.5, lw=0)

    ax.set_xlim([0.01, 1])
    ax.set_xticks([0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
    if dataset == 'cifar':
        dataset += '10'
    ax.set_title(dataset.upper())
    ax.set_ylabel("Accuracy $(\%)$")
    ax.set_xlabel('Fraction of Training Data')
    ax.grid(linestyle= 'dotted')


def main():
    params = initialize_plot('README01')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)
    x, y = set_size(398,
                    subplots=(1, 2),  # specify subplot layout for nice scaling
                    fraction=1)

    # fig = plt.figure(figsize=(x, y))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x*0.5, y*2))

    models = ['resnet50', 'dino_single', 'dino_multi']
    for idx, dataset in enumerate(['kmnist']):
        df = pd.read_csv(os.path.join(performance_log_path, f'{dataset}_logs.csv'))
        plot_performance(models, dataset, df, ax)

    # ax[1].set_ylabel('')

    # handles, labels = ax.get_legend_handles_labels()
    #
    # fig.legend(handles, labels,
    #            loc='outside lower center',
    #            # loc='best',
    #            ncol=5,
    #            columnspacing=1,
    #            # mode='expand',
    #            # bbox_to_anchor=(0, ),
    #            borderaxespad=0.5)
    # fig.show()
    fig.subplots_adjust(bottom=0.22, top=0.92, left=0.15, right=0.90)
    # fig.show()
    plt_folder_path = get_plots_subfolder_path('fmv_performance')
    plt_save_path = os.path.join(plt_folder_path, f'fmv_performance')
    plt.savefig(f'{plt_save_path}.pgf', format='pgf', backend='pgf')
    plt.savefig(f'{plt_save_path}.png', format='png', dpi=300)
    plt.savefig(f'{plt_save_path}.pdf', format='pdf', backend='pdf')


    # plt.grid()
    plt.show()
    pass


if __name__ == '__main__':
    main()