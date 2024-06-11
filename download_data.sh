#!/bin/bash
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
# wget https://storage.googleapis.com/europar_data_bsa/data.tar.gz


tar -xvf dlmc.tar.gz
# tar -xvzf data.tar.gz

mv dlmc data/
mv data/2048_512_dlmc_data.txt data/dlmc/


MTX_DIR=data/dlmc_mtx

mkdir -p ${MTX_DIR}
python3.9 utils/transform_smtx2mtx.py -i data/dlmc -o ${MTX_DIR}