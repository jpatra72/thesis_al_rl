import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from viz_utils.calculate_stuff import calculate_gp_posterior
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors

from tools.helper import plots_path, get_plots_subfolder_path, logs_path

tsne_logs = '/Users/washington/Desktop/3.5 DLR/Github/ALPBNN/logs/tsne_exp'
dataset_classes = {
    "mnist": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "kmnist": ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'],
    "cifar": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}
c = load_colors()
c10_keys = [key for key in c if '75' in key]


def tsne_plots(dataset, perplexity):
    models = ['DINO', 'RESNET50']
    tsne_result = [np.load(os.path.join(tsne_logs, f'npdata_cluster_{perplexity}',  f'{model.lower()}_{dataset}.npz')) for model in models]
    tsne_emb = []
    labels = []
    for res in tsne_result:
        tsne_emb.append(res['tsne_result'])
        labels.append(res['y_test'])

    classes = dataset_classes[dataset]

    c = load_colors()
    if 'DINO' in models:
        idx = models.index('DINO')
        models[idx] = 'DINOv2'
    if dataset == 'cifar':
        dataset += '10'

    # load params
    params = initialize_plot('README01')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)

    x, y = set_size(398,
                    subplots=(1, 2),  # specify subplot layout for nice scaling
                    fraction=1.)  # scale width/height
    # fig = plt.figure(figsize=(x, y))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(x, y * 1.5))
    fig.tight_layout()

    legend_handles = []
    # for e, l in zip(tsne_emb, labels):
    for idx, ax in enumerate([ax1, ax2]):
        for l in range(10):
            indices = [i for i, x in enumerate(labels[idx]) if x == l]
            ax.scatter(tsne_emb[idx][indices, 0], tsne_emb[idx][indices, 1], label=classes[l],
                       color=c[c10_keys[l]],
                       s=0.75,
                       alpha=0.65)
            ax.set_title(f'{models[idx]}')
            ax.set_xticks([])
            ax.set_yticks([])
            # legend_handles.append(scatter)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='outside lower center',
               bbox_to_anchor=(0.5, 1.03),
               # loc='best',
               ncol=5,
               columnspacing=1,
               # mode='expand',
               # bbox_to_anchor=(0, ),
               borderaxespad=0)
    # plt.title()
    # plt.legend()
    # fig.subplots_adjust(bottom=0.17, top=0.90, left=0.04, right=0.96)
    # plt.show()
    # plt.savefig('test_100.png', format='png', dpi=300)
    plt_folder_path = get_plots_subfolder_path('tsne_embeddings')
    plt_save_path = os.path.join(plt_folder_path, f'{dataset}_{perplexity}')
    # plt.savefig(f'{plt_save_path}.pgf', format='pgf', backend='pgf')
    # plt.savefig(f'{plt_save_path}.png', format='png', dpi=300)
    plt.savefig(f'{plt_save_path}.pdf', format='pdf', backend='pdf',
                bbox_inches='tight', pad_inches=0.05)
    plt.show()



def performance_plots():

    pass


if __name__ == '__main__':
    #
    for p in [50]:
        for d in ['cifar' , 'mnist', 'kmnist']:
            tsne_plots(dataset=d, perplexity=p)



