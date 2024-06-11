from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
import code
import numpy as np
from functools import partial
import warnings

PRJ_DIR = os.getenv("PRJ_DIR")

RESULT_DIR = f"{PRJ_DIR}/result"
FNAME = "suitesparse_reorder.csv"
SAVE_FIG_FNAME = f"{PRJ_DIR}/plots/pdf_fig_6.pdf"

PLOT_ALPHAS = [0.1, 0.3, 0.5,  0.7, 0.9]
SPMAT_ROW_EXPONENT = [10, 11, 12, 13]

NUM_ROWS = [1000, 2000, 4000, 10000]
BAR_WIDTH = 0.15


def configured_plt(df):
    plt.rcParams.update({'font.size': 15})
    fig, time_ax = plt.subplots()
    clstr_ax = time_ax.twinx()

    time_ax.set_yscale('log', base=2)
    clstr_ax.set_yscale('log', base=2)
    time_ax.set_xlabel("# of rows", fontsize=19)
    time_ax.set_ylabel('Avg. elapsed time (ms)', fontsize=19)

    clstr_ax.set_ylabel('Avg. # of clusters', fontsize=19)
    clstr_ax.set_ylim([4, df['cluster_cnt'].max()*4])
    time_ax.set_ylim([1, df['cluster_cnt'].max()*4])
    plt.xticks([i for i in range(len(NUM_ROWS))],
               [i for i in NUM_ROWS])
    # plt.legend(loc='lower center')
    # ax2.legend(title="# of clusters when the # of row is", loc='lower center', bbox_to_anchor=(0.5,-0.15),  ncol=len(ROW_EXPONENT))
    return fig, time_ax, clstr_ax


def plot_reordering_bar(plotting_df, fig_ax):
    count = 0

    color = iter(cm.viridis_r(np.linspace(0, 1, len(PLOT_ALPHAS))))
    for idx, alpha in enumerate(PLOT_ALPHAS):
        tmp = plotting_df[plotting_df.alpha == alpha].sort_values(by="rows")
        ary_elapsed = np.array(tmp.avg_reordering_time)
        x_offset = np.ones(len(ary_elapsed)) * (-1*len(PLOT_ALPHAS) /
                                                2+count) * BAR_WIDTH + np.arange(len(ary_elapsed))
        # code.interact(local=locals())
        if (len(PLOT_ALPHAS) % 2 == 1):
            x_offset = x_offset + np.ones(len(ary_elapsed)) * BAR_WIDTH/2

        c = next(color)
    #     print(ary_elapsed)
        fig_ax.bar(x_offset, ary_elapsed, label=r"$\alpha$ = {}".format(
            alpha), width=BAR_WIDTH, color=c, zorder=-1)
        if idx == 0:
            x_start = x_offset[0]
            y_start = ary_elapsed[0]
        elif idx == len(PLOT_ALPHAS) - 1:
            x_end = x_offset[0]
            y_end = ary_elapsed[0]
            plt.text(x_end-0.25, y_end + 60, fr"{(y_end) / (y_start): .1f}$\times$",
                     color="red", fontsize=14, linespacing=0.05, weight="bold")
            fig_ax.annotate("", (x_end+0.1, y_end), (x_start-0.1, y_start),
                            arrowprops=dict(arrowstyle="<->", linewidth=2, color="red"))

        count += 1


def plot_numcluster_line(plotting_df, fig_ax):
    count = 0
    color = iter(cm.Dark2(np.linspace(0, 1, len(NUM_ROWS))))
    for idx, row_size in enumerate(NUM_ROWS):
        tmp = plotting_df[plotting_df.rows == row_size].sort_values(by="alpha")
        tmp = tmp[tmp.alpha.isin(PLOT_ALPHAS)]
        ary_num_cluster = np.array(tmp.cluster_cnt)
        # ary_num_compare = np.array(tmp.num_compare)
        x_offset = count + BAR_WIDTH * \
            np.arange(len(ary_num_cluster)) - len(PLOT_ALPHAS) * BAR_WIDTH/2
        if (len(PLOT_ALPHAS) % 2 == 1):
            x_offset = x_offset + np.ones(len(PLOT_ALPHAS)) * BAR_WIDTH/2

        c = next(color)
        print(ary_num_cluster)
        fig_ax.plot(x_offset, ary_num_cluster, linestyle='--',
                    marker='o', c=c, label="{}".format(row_size), zorder=-1)
        if idx == 0:
            x_start = x_offset[0]
            y_start = ary_num_cluster[0]
            x_end = x_offset[len(ary_num_cluster) - 1]
            y_end = ary_num_cluster[len(ary_num_cluster) - 1]
            plt.text(x_end+0.05, y_end - 140, fr"{(y_end) / (y_start): .0f}$\times$",
                     color="blue", fontsize=14, linespacing=0.05, weight="bold")
            fig_ax.annotate("", (x_end, y_end), (x_start, y_start), arrowprops=dict(
                arrowstyle="<->", linewidth=2, color="blue"))

        count += 1


if __name__ == '__main__':
    file = os.path.join(RESULT_DIR, FNAME)
    assert (os.path.isfile(file))
    import pandas
    df = pd.read_csv(file)
    df = df[df.rows.isin(NUM_ROWS)]
    df = df[df.alpha.isin(PLOT_ALPHAS)]

    records = []
    warnings.filterwarnings(action='ignore')
    for alpha in PLOT_ALPHAS:
        for nrows in NUM_ROWS:
            df_tmp = df[df.rows == nrows][df.alpha == alpha]
            records.append({'rows': nrows,
                            'alpha': alpha,
                            'num_matrices': len(df_tmp),
                            'avg_reordering_time': df_tmp.avg_reordering_time.mean(),
                            'cluster_cnt': df_tmp.cluster_cnt.mean()})
    plot_data = DataFrame(records)

    warnings.filterwarnings(action='default')

    print(plot_data)

    fig, t_ax, c_ax = configured_plt(plot_data)
    plot_reordering_bar(plot_data, t_ax)
    plot_numcluster_line(plot_data, c_ax)

    for f in t_ax.texts:
        fig.texts.append(f)
    for f in c_ax.texts:
        fig.texts.append(f)
    t_ax.legend(labelspacing=0.16, handlelength=1.8,
                fontsize=13, loc='upper left')

    plt.savefig(SAVE_FIG_FNAME,
                format="pdf", bbox_inches="tight")
