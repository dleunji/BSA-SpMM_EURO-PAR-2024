#!/bin/bash

# 1-SA
cd baselines/1-sa
make clean
make cuda

# sputnik
cd ../sputnik
make clean
bash setup.sh

# TC-GNN ()
cd ../tc-gnn
bash 0_build_tcgnn.sh

# BSA-SpMM
cd ../..
make clean
make all
