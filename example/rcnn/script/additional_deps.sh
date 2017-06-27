#!/usr/bin/env bash

# install additional depts
sudo apt install python-pip python-dev unzip python-matplotlib
sudo pip install cython scikit-image easydict

# install a forked MXNet
pushd ../../
cp make/config.mk ./
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk
make -j$(nproc)
pushd python
python setup.py install --user
popd
popd

# build cython extension
make
