#!/bin/bash
set -e
sudo apt-get update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt install -y \
	build-essential \
	git \
	autoconf \
	libtool \
	unzip \
	gcc-4.8 \
	g++-4.8 \
	gfortran \
	gfortran-4.8 \
	nasm \
	make \
	automake \
	pkg-config \
	pandoc \
	python-dev \
	libssl-dev \
	python-pip

wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz
tar -xvzf cmake-3.12.3.tar.gz
cd cmake-3.12.3
./bootstrap
make -j
sudo make install

pip install -U pip "setuptools==36.2.0" wheel --user
pip install pypandoc numpy==1.15.0 --user

export mxnet_variant=mkl

git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet

bash tools/staticbuild/build.sh

mv python/setup.py python/setup.py.bak
cp tools/pip/setup.py python/
cp tools/pip/MANIFEST.in python/
cp -r tools/pip/doc python/

bash tools/staticbuild/build_wheel.sh
