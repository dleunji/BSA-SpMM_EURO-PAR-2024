# TORCH_CUDA_ARCH_LIST="8.6" python setup.py install
cd TCGNN_conv
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
python3.9 setup.py install
cd ..