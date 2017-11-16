#!/usr/bin/env bash

set -e

yum groupinstall -y "Development Tools"
yum install -y cmake python27 python27-setuptools
git clone https://github.com/opencv/opencv
cd opencv
mkdir -p build
cd build
cmake -D BUILD_opencv_gpu=OFF -D WITH_EIGEN=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make PREFIX=/usr/local install

