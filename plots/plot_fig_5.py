import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
import os

PRJ_DIR = os.getenv("PRJ_DIR")

NROWS = 2048
NCOLS = 512

def preprocess_df(df, delta):
    df["sparsity"] = df["matrix"].apply(lambda x: 100 * float(x.split("/")[-2]))
    df = df.loc[df["delta"] == delta]
    df1 = df.loc[(df["rows"] == NCOLS) & (df["cols"] == NROWS)]
    df2 = df.loc[(df["rows"] == NROWS) & (df["cols"] == NCOLS)]
    df = pd.concat([df1, df2])
    
    df = df.loc[df.groupby("matrix")["avg_density_of_tiles"].idxmax()]
    df = df.sort_values(by="matrix")
    df.reset_index(inplace=True)
    
    return df

def compare_reordering(bsa, non, delta):
    bsa = preprocess_df(bsa, delta)
    non = preprocess_df(non, delta)
    
    bsa["diff_num_tiles"] = list(bsa["num_tiles"] - non["num_tiles"])
    bsa["diff_avg_density"] = list(bsa["avg_density_of_tiles"] - non["avg_density_of_tiles"])
    
    return bsa
  
if __name__ == "__main__":
    bsa_data = pd.read_csv(f"{PRJ_DIR}/result/reordering-dlmc-bsa.csv")
    non_data = pd.read_csv(f"{PRJ_DIR}/result/reordering-dlmc-non.csv")
    deltas = [0.2, 0.4, 0.6, 0.8]
    for delta in deltas:
            plot_data = compare_reordering(bsa_data, non_data, delta)
            x_value = plot_data["diff_num_tiles"]
            y_value = plot_data["diff_avg_density"]
            plt.scatter(x_value, y_value, 8, label=r"$\delta=${}".format(delta), marker="o")
    kwargs = {"linewidth":"1"}
    plt.axvline(x=0,linestyle="dashed",color="black", **kwargs)
    plt.axhline(y =0, linestyle="dashed", color="black",**kwargs)
    # plt.yscale("log")

    # plt.xscale("log")
    plt.xticks(fontsize=13)
    ytick = np.arange(0, 0.91, 0.1)
    plt.yticks(ytick, fontsize=13)
    plt.ylabel(r"$\Delta$ Avg. density of dense tiles", fontsize=18)
    plt.xlabel(r"$\Delta$ # of dense tiles", fontsize=18)
    plt.legend(fontsize=13, labelspacing=0.5, handlelength=0.6, loc="upper left")
    plt.savefig(f"{PRJ_DIR}/plots/pdf_fig_5.pdf", format="pdf", bbox_inches="tight")