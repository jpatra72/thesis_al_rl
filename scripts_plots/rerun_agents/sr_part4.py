import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from viz_utils.calculate_stuff import calculate_gp_posterior
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors

from tools.helper import plots_path, get_plots_subfolder_path, logs_path

sr_log_path = {'sr3': "/Users/washington/Desktop/3.5 DLR/Github/ALPBNN/logs/sr3",
               'sr4': "/Users/washington/Desktop/3.5 DLR/Github/ALPBNN/logs/sr4"}
c = load_colors()
ckey_dict = {'dqn0': 'blau',
             'dqn1': 'gruen',
             'entropy': 'magenta',
             'margin': 'orange',
             'rstream': 'schwarz',
             'random': 'tuerkis'}

ckey_list = ['gruen', 'schwarz', 'blau', 'maigruen', 'tuerkis', 'orange', ]
c_100_dict = {key: f'{val}100' for key, val in ckey_dict.items()}
c_75_dict = {key: f'{val}75' for key, val in ckey_dict.items()}
c_50_dict = {key: f'{val}50' for key, val in ckey_dict.items()}
c_25_dict = {key: f'{val}25' for key, val in ckey_dict.items()}

# labels_dict = {'dqn0': 'sr1',
#                'dqn1': f'sr1 without $s^{a}$',
#                'entropy': 'entropy',
#                'coreset': 'coreset',
#                'rstream': 'rand-stream',
#                'random': 'rand'}
baselines_len = 81
dqn_len = 401


def find_key(input_dict, value):
    result = "None"
    for key, val in input_dict.items():
        if val == value:
            result = key
    return result


def plot_performance(df_dqn, df_rstream, df_baselines, baselines, ax, plot_order, dataset, budget=200, skip=10,
                     std=True):
    performance_mean = {}
    performance_max = {}
    performance_min = {}
    step = {}

    b_cnt = budget // 5 + 1
    dqn_cnt = budget + 1

    b_skip = skip // 5
    dqn_skip = skip

    df_baselines_cnames = df_baselines.columns.tolist()
    # df_baselines_cnames = [cname.lower() for cname in df_baselines_cnames]
    for baseline in baselines:
        mean_cname = [cname for cname in df_baselines_cnames
                      if baseline in cname.lower() and all(
                keyword not in cname.lower() for keyword in ['step', 'min', 'max'])]
        min_cname = [cname for cname in df_baselines_cnames
                     if baseline in cname.lower() and 'min' in cname.lower() and 'step' not in cname.lower()]
        max_cname = [cname for cname in df_baselines_cnames
                     if baseline in cname.lower() and 'max' in cname.lower() and 'step' not in cname.lower()]
        step_cname = [cname for cname in df_baselines_cnames
                      if 'global_step' in cname.lower()]
        performance_mean[baseline] = df_baselines[mean_cname[0]].tolist()[0:b_cnt:b_skip]
        performance_max[baseline] = df_baselines[max_cname[0]].tolist()[0:b_cnt:b_skip]
        performance_min[baseline] = df_baselines[min_cname[0]].tolist()[0:b_cnt:b_skip]
        step[baseline] = df_baselines[step_cname[0]].tolist()[0:b_cnt:b_skip]

    mean_cname = [cname for cname in df_dqn.columns.tolist() if
                  'min' not in cname.lower() and 'max' not in cname.lower() and 'step' not in cname.lower()]
    max_cname = [cname for cname in df_dqn.columns.tolist() if 'max' in cname.lower() and 'model' in cname.lower()]
    min_cname = [cname for cname in df_dqn.columns.tolist() if 'min' in cname.lower() and 'model' in cname.lower()]
    step_cname = [cname for cname in df_dqn.columns.tolist()
                  if 'global_step' in cname.lower()]


    if 'sr' not in mean_cname[0]:
        mean_cname =  list(reversed(mean_cname))
        max_cname = list(reversed(max_cname))
        min_cname = list(reversed(min_cname))
        step_cname = list(reversed(step_cname))

    for i in range(2):
        performance_mean[f'dqn{i}'] = df_dqn[mean_cname[i]].tolist()[0:dqn_cnt:dqn_skip]
        performance_max[f'dqn{i}'] = df_dqn[max_cname[i]].tolist()[0:dqn_cnt:dqn_skip]
        performance_min[f'dqn{i}'] = df_dqn[min_cname[i]].tolist()[0:dqn_cnt:dqn_skip]
        step[f'dqn{i}'] = df_dqn[step_cname[0]].tolist()[0:dqn_cnt:dqn_skip]


    ax.axvline(x=200, c='grey', lw=0.75, linestyle='--')

    for idx, p in enumerate(plot_order):
        ax.plot(step[p], performance_mean[p], color=c[c_100_dict[p]], label=labels_dict[plot_order[idx]],
                linewidth=0.75, alpha=0.7)
        if std == True:
            ax.fill_between(step[p], performance_min[p], performance_max[p],
                            color=c[c_50_dict[p]], alpha=0.4, lw=0)

    ax.set_xlim([0, budget])
    ax.set_ylabel("Classifier's Accuracy $(\%)$")
    ax.set_xlabel('Labeled Count')
    # ax.set_title(r"MNIST $\rightarrow$ " f"{dataset.upper()}")
    ax.set_title(f"{dataset.upper()}")
    ax.grid(linestyle='dotted')
    # plt.show()
    pass


def main(state, budget, skip, std):
    params = initialize_plot('README01')
    plt.rcParams.update(params)
    x, y = set_size(398,
                    subplots=(1, 2),  # specify subplot layout for nice scaling
                    fraction=1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(x, y * 1.3))
    fig.tight_layout()
    ax = ax.reshape(-1)

    # state = 'sr2'

    baselines = ['margin']

    plot_order = ['dqn0', 'dqn1', 'margin']

    for idx, dataset in enumerate(['mnist', 'kmnist']):
        df_dqn0 = pd.read_csv(os.path.join(sr_log_path[state], f'{dataset.lower()}.csv'))
        df_dqn1 = pd.read_csv(os.path.join(sr_log_path[state], f'{dataset.lower()}.csv'))
        df_baselines = pd.read_csv(os.path.join(sr_log_path[state], f'baselines_{dataset.lower()}.csv'))

        plot_performance(df_dqn0, df_dqn1, df_baselines, baselines, ax[idx], plot_order, dataset, budget, skip, std)

    ax[1].set_ylabel('')

    handles, labels = ax[1].get_legend_handles_labels()

    legend_handles = [plt.Line2D([0], [0],
                                 linewidth=1, alpha=0.7,
                                 label=label,
                                 color=c[c_100_dict[find_key(labels_dict, label)]]) for label in labels]

    fig.legend(legend_handles, labels,
               loc='outside lower center',
               bbox_to_anchor=(0.5, 1.03),
               ncol=6,
               columnspacing=1,
               borderaxespad=0)
    # leg = plt.legend()
    # leg_lines = leg.get_lines()
    # plt.setp(leg_lines, linewidth=2)

    plt_folder_path = get_plots_subfolder_path(state)
    plt_save_path = os.path.join(plt_folder_path, f'{state}_{budget}_{skip}_{int(std)}')
    plt.savefig(f'{plt_save_path}.pdf', format='pdf', backend='pdf',
                bbox_inches='tight', pad_inches=0.05
                )
    plt.show(bbox_inches='tight', pad_inches=0.05)
    pass

state = 'sr4'
a = 'SR1' if state == 'sr3' else 'SR2'
labels_dict = {'dqn0': f"{a}",
               'dqn1': f"{a}'",
               'margin': 'margin',
               'coreset': 'coreset',
               'rstream': 'rand-stream',
               'random': 'rand'}

if __name__ == '__main__':
    # state = 'sr4'
    for b in [400]:
        for std in [True, False]:
            # skip = 5 if b == 200 or b == 300 else 10
            skip = 10
            main(state, b, skip, std)
    # main()
