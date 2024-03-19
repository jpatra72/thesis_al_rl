import os

import matplotlib.pyplot as plt
import pandas as pd
from tools.helper import get_plots_subfolder_path
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors

sr_log_path = {'sr1': "/Users/washington/Desktop/3.5 DLR/Github/ALPBNN/logs/sr1",
               'sr2': "/Users/washington/Desktop/3.5 DLR/Github/ALPBNN/logs/sr2"}
c = load_colors()
ckey_dict = {'final': 'blau',
             'margin': 'gruen',
             'entropy': 'magenta',
             'coreset': 'orange',
             'eps0': 'schwarz',
             'random': 'tuerkis'}

ckey_list = ['gruen', 'schwarz', 'blau', 'maigruen', 'tuerkis', 'orange', ]
c_100_dict = {key: f'{val}100' for key, val in ckey_dict.items()}
c_75_dict = {key: f'{val}75' for key, val in ckey_dict.items()}
c_50_dict = {key: f'{val}50' for key, val in ckey_dict.items()}
c_25_dict = {key: f'{val}25' for key, val in ckey_dict.items()}

labels_dict = {'final': 'trained agent',
               'margin': 'margin',
               'entropy': 'entropy',
               'coreset': 'coreset',
               'eps0': 'episode 0',
               'random': 'random'}
# baselines_len = 81
dqn_len = 200


def find_key(input_dict, value):
    result = "None"
    for key, val in input_dict.items():
        if val == value:
            result = key
    return result


def plot_performance(df_dqn, ax, plot_order, budget=200, skip=10,
                     std=True):
    performance_mean = {}
    performance_max = {}
    performance_min = {}
    step = {}

    b_cnt = budget // 5 + 1
    dqn_cnt = budget + 1

    b_skip = skip // 5
    dqn_skip = skip

    # df_baselines_cnames = df_baselines.columns.tolist()
    # # df_baselines_cnames = [cname.lower() for cname in df_baselines_cnames]
    # for baseline in baselines:
    #     mean_cname = [cname for cname in df_baselines_cnames
    #                   if baseline in cname.lower() and all(
    #             keyword not in cname.lower() for keyword in ['step', 'min', 'max'])]
    #     min_cname = [cname for cname in df_baselines_cnames
    #                  if baseline in cname.lower() and 'min' in cname.lower() and 'step' not in cname.lower()]
    #     max_cname = [cname for cname in df_baselines_cnames
    #                  if baseline in cname.lower() and 'max' in cname.lower() and 'step' not in cname.lower()]
    #     step_cname = [cname for cname in df_baselines_cnames
    #                   if 'global_step' in cname.lower()]
    #     performance_mean[baseline] = df_baselines[mean_cname[0]].tolist()[0:b_cnt:b_skip]
    #     performance_max[baseline] = df_baselines[max_cname[0]].tolist()[0:b_cnt:b_skip]
    #     performance_min[baseline] = df_baselines[min_cname[0]].tolist()[0:b_cnt:b_skip]
    #     step[baseline] = df_baselines[step_cname[0]].tolist()[0:b_cnt:b_skip]

    mean_cname = [cname for cname in df_dqn.columns.tolist() if
                  'min' not in cname.lower() and 'max' not in cname.lower() and 'final' in cname.lower()]
    max_cname = [cname for cname in df_dqn.columns.tolist() if 'max' in cname.lower() and 'model' in cname.lower() and 'final' in cname.lower()]
    min_cname = [cname for cname in df_dqn.columns.tolist() if 'min' in cname.lower() and 'model' in cname.lower() and 'final' in cname.lower()]
    step_cname = [cname for cname in df_dqn.columns.tolist()
                  if 'global_step' in cname.lower()]
    performance_mean['final'] = df_dqn[mean_cname[0]].tolist()[0:dqn_cnt:dqn_skip]
    performance_max['final'] = df_dqn[max_cname[0]].tolist()[0:dqn_cnt:dqn_skip]
    performance_min['final'] = df_dqn[min_cname[0]].tolist()[0:dqn_cnt:dqn_skip]
    step['final'] = df_dqn[step_cname[0]].tolist()[0:dqn_cnt:dqn_skip]

    mean_cname = [cname for cname in df_dqn.columns.tolist() if
                  'min' not in cname.lower() and 'max' not in cname.lower() and 'eps0' in cname.lower()]
    max_cname = [cname for cname in df_dqn.columns.tolist() if 'max' in cname.lower() and 'model' in cname.lower() and 'eps0' in cname.lower()]
    min_cname = [cname for cname in df_dqn.columns.tolist() if 'min' in cname.lower() and 'model' in cname.lower() and 'eps0' in cname.lower()]
    step_cname = [cname for cname in df_dqn.columns.tolist()
                  if 'global_step' in cname.lower()]
    performance_mean['eps0'] = df_dqn[mean_cname[0]].tolist()[0:dqn_cnt:dqn_skip]
    performance_max['eps0'] = df_dqn[max_cname[0]].tolist()[0:dqn_cnt:dqn_skip]
    performance_min['eps0'] = df_dqn[min_cname[0]].tolist()[0:dqn_cnt:dqn_skip]
    step['eps0'] = df_dqn[step_cname[0]].tolist()[0:dqn_cnt:dqn_skip]

    # plot_order = ['dqn', 'random', 'rstream', 'entropy', 'margin', 'coreset']
    plot_order = ['eps0', 'final']
    # ax.axvline(x=200, c='grey', lw=0.75, linestyle='--')

    for idx, p in enumerate(plot_order):
        ax.plot(step[p], performance_mean[p], color=c[c_100_dict[p]], label=labels_dict[plot_order[idx]],
                linewidth=0.75, alpha=0.7)
        if std == True:
            ax.fill_between(step[p], performance_min[p], performance_max[p],
                            color=c[c_50_dict[p]], alpha=0.4, lw=0)
    ax.set_xlim([0, budget])
    ax.set_ylabel("Classifier's Accuracy $(\%)$")
    ax.set_xlabel('Labeled Count')
    ax.set_title(f"State Representation 1")
    ax.grid(linestyle='dotted')
    # plt.show()
    pass


def main(budget, skip, std):
    params = initialize_plot('README01')
    plt.rcParams.update(params)
    x, y = set_size(398,
                    subplots=(1, 2),  # specify subplot layout for nice scaling
                    fraction=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x * 0.505, y * 1.3))
    fig.tight_layout()
    # ax = ax.reshape(-1)

    state = 'sr1'

    baselines = [None]

    plot_order = ['dqn0', 'dqn1']

    for i in range(1):
        df_dqn = pd.read_csv(os.path.join(sr_log_path[state], f'agent_training.csv'))

        plot_performance(df_dqn, ax, plot_order, budget, skip, std)

    # ax[1].set_ylabel('')

    handles, labels = ax.get_legend_handles_labels()

    legend_handles = [plt.Line2D([0], [0],
                                 linewidth=1, alpha=0.7,
                                 label=label,
                                 color=c[c_100_dict[find_key(labels_dict, label)]]) for label in labels]

    fig.legend(legend_handles, labels,
               loc='outside lower center',
               bbox_to_anchor=(0.5, 1.03),
               ncol=2,
               columnspacing=1,
               borderaxespad=0)
    # leg = plt.legend()
    # leg_lines = leg.get_lines()
    # plt.setp(leg_lines, linewidth=2)

    plt_folder_path = get_plots_subfolder_path(state)
    plt_save_path = os.path.join(plt_folder_path, f'{state}_agent_training_mnist_{int(std)}')
    plt.savefig(f'{plt_save_path}.pdf', format='pdf', backend='pdf',
                bbox_inches='tight', pad_inches=0.05
                )
    plt.show(bbox_inches='tight', pad_inches=0.05)
    pass


if __name__ == '__main__':
    for b in [200]:
        for std in [True, False]:
            # skip = 5 if b == 200 or b == 300 else 10
            skip = 5
            main(b, skip, std)
    # main()
