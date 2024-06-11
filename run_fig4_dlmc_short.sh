#!/bin/bash
# 1-SA
bash ${PRJ_DIR}/scripts_short/fig4_1-sa_dlmc_short.sh
# BSA-SpMM
bash ${PRJ_DIR}/scripts_short/fig4_bsa_dlmc_short.sh
# cuBLAS
bash ${PRJ_DIR}/scripts_short/fig4_cublas_dlmc_short.sh
# cuSPARSE
bash ${PRJ_DIR}/scripts_short/fig4_cusparse_dlmc_short.sh
# Sputnik
bash ${PRJ_DIR}/scripts_short/fig4_sputnik_dlmc_short.sh
# TC-GNN
bash ${PRJ_DIR}/scripts_short/fig4_tcgnn_dlmc_short.sh

python3.9 plots/plot_fig_4.py