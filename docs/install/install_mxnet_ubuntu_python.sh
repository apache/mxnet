#!/bin/bash

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
# This script installs MXNet with required dependencies on a Ubuntu Machine.
# Tested on Ubuntu 16.04+ distro.
# Important Maintenance Instructions:
#    Align changes with CI in /ci/docker/install/ubuntu_core.sh
#    and ubuntu_python.sh
######################################################################

set -ex
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libjemalloc-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libzmq3-dev \
    ninja-build \
    python-dev \
    python3-dev \
    software-properties-common \
    sudo \
    unzip \
    virtualenv \
    wget

wget -nv https://bootstrap.pypa.io/get-pip.py
echo "Installing for Python 3..."
sudo python3 get-pip.py
pip3 install --user -r requirements.txt
echo "Installing for Python 2..."
sudo python2 get-pip.py
pip2 install --user -r requirements.txt

cd ../../

echo "Checking for GPUs..."
gpu_install=$(which nvidia-smi | wc -l)
if [ "$gpu_install" = "0" ]; then
    make_params="USE_OPENCV=1 USE_BLAS=openblas"
    echo "nvidia-smi not found. Installing in CPU-only mode with these build flags: $make_params"
else
    make_params="USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
    echo "nvidia-smi found! Installing with CUDA and cuDNN support with these build flags: $make_params"
fi

echo "Building MXNet core. This can take few minutes..."
make -j $(nproc) $make_params
