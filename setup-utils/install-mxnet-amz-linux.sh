#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

######################################################################
# This script installs MXNet for Python along with all required dependencies on a Amazon Linux Machine.
######################################################################
set -e
# CMake is required for installing dependencies.
sudo yum install -y cmake

# Set appropriate library path env variables
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.profile
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.profile
echo '. ~/.profile' >> ~/.bashrc
source ~/.profile

# Install gcc-4.8/make and other development tools on Amazon Linux
# Reference: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compile-software.html
# Install Python, Numpy, Scipy and set up tools.
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python27 python27-setuptools python27-tools python-pip graphviz
sudo yum install -y python27-numpy python27-scipy python27-nose python27-matplotlib

# Install OpenBLAS at /usr/local/openblas
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran -j $(($(nproc) + 1))
sudo make PREFIX=/usr/local install
cd ..

# Install OpenCV at /usr/local/opencv
git clone https://github.com/opencv/opencv
cd opencv
mkdir -p build
cd build
cmake -D BUILD_opencv_gpu=OFF -D WITH_EIGEN=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
sudo make PREFIX=/usr/local install

# Export env variables for pkg config
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Install MXNet Core without CUDA
MXNET_HOME="$HOME/mxnet/"
cd "$MXNET_HOME"
cp make/config.mk .
echo "USE_CUDA=0" >>config.mk
echo "USE_CUDNN=0" >>config.mk
echo "USE_BLAS=openblas" >>config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk
echo "ADD_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs" >>config.mk
sudo make -j$(nproc)

# Install MXNet Python package
cd python
sudo python setup.py install

# Add MXNet path to ~/.bashrc file
echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

# Install graphviz for visualizing network graph and jupyter notebook to run tutorials and examples
sudo pip install graphviz
sudo pip install jupyter

echo "Done! MXNet for Python installation is complete. Go ahead and explore MXNet with Python :-)"
