#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

PRJ_DIR = os.getenv("PRJ_DIR")

model = 'gcn'
hidden = 16
datasets = ["transformer", "rn50"]
sub_datasets = ["variational_dropout", "magnitude_pruning", "variational_dropout"]
DLMC_DIR = f"{PRJ_DIR}/data/dlmc"
for DATASET in datasets:
    for pruning in os.listdir(os.path.join(DLMC_DIR, DATASET)):
        pruning_path = os.path.join(DLMC_DIR, DATASET, pruning)
        if os.path.isdir(pruning_path):
            for sparsity in os.listdir(pruning_path):
                sparsity_path = os.path.join(pruning_path, sparsity)
                if os.path.isdir(sparsity_path):
                    cnt = 0
                    stop = 3
                    for file in os.listdir(sparsity_path):
                        if cnt == stop:
                            break
                        file_path = os.path.join(sparsity_path, file)
                        if os.path.isfile(file_path):
                            # EUNJI
                            os.system(f"{PRJ_DIR}/.venv/bin/python {PRJ_DIR}/baselines/tc-gnn/dlmc_main_tcgnn.py --dataset {file_path} --epochs 2 --dim 128 --single_kernel")
                            cnt += 1 