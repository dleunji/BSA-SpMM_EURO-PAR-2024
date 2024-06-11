import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
import os

PRJ_DIR = os.getenv("PRJ_DIR")

sub_datasets = [
    "variational_dropout",
    "magnitude_pruning",
    "random_pruning"
]

def preprocess_1_sa_data(data, b_cols):
    data["dataset"] = data["matrix"].apply(lambda x: x.split("/")[-4])
    data["sub_dataset"] = data["matrix"].apply(lambda x: x.split("/")[-3])
    data["sparsity"] = data["matrix"].apply(lambda x: 100 * float(x.split("/")[-2]))
    data = data.loc[data["sparsity"] <= 90.0 ]
    data = data.loc[data["b_cols"] == b_cols]
    data = data.astype({'sparsity':'int32'})
    data = data.loc[data.groupby("matrix")["avg_time_multiply"].idxmin()]
    data = data.rename(columns = {"avg_time_multiply": "avg_total_time"})
    data = data.sort_values(by=["matrix"])
    data.reset_index(inplace=True)
    return data
      
def preprocess_dafault_data(data, b_cols):
    data["dataset"] = data["matrix"].apply(lambda x: x.split("/")[-4])
    data["sub_dataset"] = data["matrix"].apply(lambda x: x.split("/")[-3])
    data["sparsity"] = data["matrix"].apply(lambda x: 100 * float(x.split("/")[-2]))
    data = data.loc[data["sparsity"] <= 90.0 ]
    data = data.loc[data["n_cols"] == b_cols]
    data = data.astype({'sparsity':'int32'})
    data = data.loc[data.groupby("matrix")["avg_total_time"].idxmin()]
    data = data.sort_values(by=["matrix"])
    data.reset_index(inplace=True)
    return data
  
# tc-gnn, aspt-rr, sputnik
def preprocess_baseline_data(data):  
#      data = root_data.copy(deep=True)
    data["dataset"] = data["matrix"].apply(lambda x: x.split("/")[-4])
    data["sub_dataset"] = data["matrix"].apply(lambda x: x.split("/")[-3])
    data["sparsity"] = data["matrix"].apply(lambda x: 100 * float(x.split("/")[-2]))
    data = data.loc[data["sparsity"] <= 90.0 ]
    data = data.astype({'sparsity':'int32'})
    data = data.sort_values(by=["matrix"])
    data.reset_index(inplace=True)
    return data

def extract_dataset(data, dataset, sub_dataset):
        data = data.loc[data["dataset"] == dataset]
        data = data.loc[data["sub_dataset"] == sub_dataset]
        return data

def calculate_base_pos(sparsity_list):
    X = np.arange(len(sparsity_list), dtype=float)
    X[0] = 0
    X[len(sparsity_list) - 1] = 5
    for idx in range(1, len(sparsity_list)):
        X[idx] = (5 / (len(sparsity_list) - 1)) * idx
    return X
  
def calculate_pos(base_pos, len_p, w_):
    pos_list = []
    for i in range(0, len_p):
            pos = [idx + i * w_ for idx in base_pos]
            pos_list.append(pos)
    return pos_list
  
# meta-data
# 0. cublas
# 1. cusparse
# 2. aspt-rr
# 3. sptunik
# 4. 1-sa
# 5. tc-gnn
# 6. bsa

colors = ["", "plum", "skyblue", "limegreen", "royalblue", "orange", "tomato"]
# median_dict = dict(markeredgecolor="black", markeredgewidth=1.5)
median_dict = dict(linestyle='-', linewidth=1.8, color='black')

def plot(idx, dataset, sub_dataset, pos, plot_data, axis, row, col):
    boxprops_ = dict(linewidth=1, color=colors[idx], facecolor=colors[idx])
    cap_dict = dict(color=colors[idx])
    flier_dict = dict(markeredgecolor=colors[idx])
    if idx == 4:
            # ticklabel O
            boxplot = axis[row][col].boxplot(plot_data.values, labels=plot_data.index,positions=pos,
                                    manage_ticks=True,patch_artist=True, notch=False, vert=True,
                                    meanline=False,showmeans=False, showfliers=False,
                                    medianprops = median_dict, flierprops=flier_dict,capprops=cap_dict,whiskerprops=cap_dict,
                                    widths=W_,boxprops=boxprops_)
    else:
            # ticklabel X
            boxplot = axis[row][col].boxplot(plot_data.values, labels=plot_data.index,positions=pos,
                                    manage_ticks=False,patch_artist=True, notch=False, vert=True,
                                    meanline=False,showmeans=False, showfliers=False,
                                    medianprops = median_dict, flierprops=flier_dict,capprops=cap_dict,whiskerprops=cap_dict,
                                    widths=W_,boxprops=boxprops_)
    axis[row][col].tick_params(labelsize=18)
    axis[row][col].set_title(f"{sub_dataset} ({dataset})", fontsize=23, pad=15)
    # XXX
    # axis[row][col].set_ylim(0, 5)

    return boxplot
  

if __name__ == "__main__":
    b_cols = 128
    datasets = ["transformer", "rn50"]
    cased_datasets = ["Transformer", "ResNet-50"]
    # datasets = ["transformer"]

    bsa_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-bsa.csv")
    cusparse_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-cusparse.csv")
    cublas_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-cublas.csv")
    sputnik_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-sputnik.csv")
    one_sa_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-1-sa.csv")
    tcgnn_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-tcgnn.csv")
    # aspt_rr_data = pd.read_csv(f"{PRJ_DIR}/result/dlmc-aspt-rr.csv")

    pre_1_sa_data = preprocess_1_sa_data(one_sa_data, b_cols)
    pre_bsa_data = preprocess_dafault_data(bsa_data, b_cols)
    pre_cusparse_data = preprocess_dafault_data(cusparse_data, b_cols)
    pre_cublas_data = preprocess_dafault_data(cublas_data, b_cols)
    pre_tcgnn_data = preprocess_baseline_data(tcgnn_data)
    # pre_aspt_rr_data = preprocess_baseline_data(aspt_rr_data)
    pre_sputnik_data = preprocess_baseline_data(sputnik_data)

    # pre_tcgnn_data["avg_total_time"].isnull().values.any()
    
    figure, axis = plt.subplots(len(datasets), len(sub_datasets), constrained_layout=True, figsize=(20, 8))
    names = ["cuSPARSE", "Sputnik", "1-SA w/ TCs", "TC-GNN w/ TCs", "BSA-SpMM w/ TCs (our approach)"]
    total_data = [pre_cublas_data, pre_cusparse_data, pre_sputnik_data, pre_1_sa_data, pre_tcgnn_data, pre_bsa_data]

    final_data = []
    plots = []
    for i, dataset in enumerate(datasets):
            for j, sub_dataset in enumerate(sub_datasets):
                    baseline = axis[i][j].axhline(y = 1, color = 'black', linestyle = 'dotted')
                    final_data = []
                    for p, perf in enumerate(total_data):
                            ext_data = extract_dataset(total_data[p], dataset, sub_dataset)
                            final_data.append(ext_data)
                    # assert
                    for idx in range(1, len(final_data)):
                            assert(len(final_data[idx - 1]) == len(final_data[idx]))
                            
                    # calculate speedup
                    for idx in range(1, len(final_data)):
                            assert len(final_data[idx]) == len(final_data[0])
                            final_data[idx]["baseline"] = list(final_data[0]["avg_total_time"])
                            final_data[idx]["speedup"] = final_data[idx]["baseline"] / final_data[idx]["avg_total_time"]
                    # display(final_data[5][["speedup", "avg_total_time", "baseline"]])
                    
                    # print("len", len(final_data[5].loc[final_data[5]["avg_total_time"] == 0]))
                    sparsity_list = final_data[0]["sparsity"].unique()
                    W_ = 0.16
                    X = calculate_base_pos(sparsity_list)
                    pos_list = calculate_pos(X, len(final_data) - 1, W_)
                    for idx in range(1, len(final_data)):
                            pos = pos_list[idx - 1]
                            plot_data = final_data[idx].groupby("sparsity")["speedup"].apply(list)
    #                         display(plot_data)
                            boxplot = plot(idx, cased_datasets[i], sub_dataset, pos, plot_data, axis, i, j)
                            if i == 0 and j == 0:
                                    plots.append(boxplot["boxes"][0])
    figure.legend(plots, names, bbox_to_anchor=(0.5, 1.1), ncols=6, fontsize=20, loc="outside upper center")
    figure.supxlabel(r'Sparsity (%)', fontsize=22)
    figure.supylabel('Speedup over cuBLAS w/ TCs', fontsize=22)

    plt.savefig(f"{PRJ_DIR}/plots/pdf_fig_4.pdf", format="pdf", bbox_inches="tight")