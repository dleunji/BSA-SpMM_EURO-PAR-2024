#!/bin/bash
TC_GNN_DIR="${PRJ_DIR}/baselines/tc-gnn"
RESULT_DIR="${PRJ_DIR}/result"
${TC_GNN_DIR}/2_dlmc_tcgnn_single_kernel.py| tee ${TC_GNN_DIR}/dlmc-tcgnn.log 2> ${TC_GNN_DIR}/dlmc-tcgnn.err
${TC_GNN_DIR}/1_log2csv.py ${TC_GNN_DIR}/dlmc-tcgnn.log
mv ${TC_GNN_DIR}/dlmc-tcgnn.csv ${RESULT_DIR}
# ./2_tcgnn_single_kernel.py