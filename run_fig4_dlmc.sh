#!/bin/bash
# 1-SA
bash ${PRJ_DIR}/scripts/fig4_1-sa_dlmc.sh
# BSA-SpMM
bash ${PRJ_DIR}/scripts/fig4_bsa_dlmc.sh
# cuBLAS
bash ${PRJ_DIR}/scripts/fig4_cublas_dlmc.sh
# cuSPARSE
bash ${PRJ_DIR}/scripts/fig4_cusparse_dlmc.sh
# Sputnik
bash ${PRJ_DIR}/scripts/fig4_sputnik_dlmc.sh
# TC-GNN
bash ${PRJ_DIR}/scripts/fig4_tcgnn_dlmc.sh

python3.9 plots/plot_fig_4.py