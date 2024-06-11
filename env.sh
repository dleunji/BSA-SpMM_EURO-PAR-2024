#!/bin/bash

########### Change the value according to your computer ###########
export CUDA_PATH=/usr/local/cuda-12.1
export CUDA_ARCH=86
###################################################################
export PRJ_DIR=$(pwd)
export SPUTNIK_PATH=${PRJ_DIR}/sputnik/install
export PATH=${CUDA_PATH}/bin:$PATH
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${SPUTNIK_PATH}/lib:${LD_LIBRARY_PATH}"
export NCU_PATH=${CUDA_PATH}/bin/ncu


sudo apt-get install python3.9
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python3.9 get-pip.py

# sudo apt-get install python3.9-venv
sudo apt-get install python3.9-dev

if [ ! -d .venv ]; then
    # sudo apt-get install python3-venv
    python3.9 -m venv .venv
fi

source .venv/bin/activate

pip3.9 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip3.9 install pandas numpy matplotlib tqdm scipy 

# export PRJ_DIR=$(pwd)
# export CUDA_ARCH=86