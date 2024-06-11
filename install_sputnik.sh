#!/bin/bash
# install glog
# git clone https://github.com/google/glog.git
# cd glog
# mkdir build && cd build
# cmake ..
# sudo make install 

# cd ../..
sudo apt install libgoogle-glog-dev


# sputnik
git clone https://github.com/google-research/sputnik
cd sputnik
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS=${CUDA_ARCH} -DCMAKE_INSTALL_PREFIX=${PRJ_DIR}/sputnik/install
make -j12
sudo make install
