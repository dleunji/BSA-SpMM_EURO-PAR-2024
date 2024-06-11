#!/bin/bash
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz

tar -xvf dlmc.tar.gz

mv dlmc data/
mv data/2048_512_dlmc_data.txt data/dlmc/