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
######################################################################

PERL_ROOT=${HOME}
JOBS=$(nproc)

while getopts ':p:j:' o
do case "$o" in
    p) PERL_ROOT="$OPTARG";;
    j) JOBS="$OPTARG";;
    [?]) print >&2 "Usage: $0 [-p <perl root directory>] [-j <number of jobs to run simultaneously>"
	exit 1;;
    esac
done
shift "$(($OPTIND -1))"

# this list based on python install script and maybe can be reduced
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
    software-properties-common \
    sudo \
    unzip \
    virtualenv \
    wget \
    libmouse-perl \
    pdl \
    cpanminus \
    swig \
    libgraphviz-perl

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
make -j $JOBS $make_param

echo "Install perl interface..."

MXNET_HOME=${PWD}
export LD_LIBRARY_PATH=${MXNET_HOME}/lib
export PERL5LIB=${PERL_ROOT}/perl5/lib/perl5

cpanm -q -L "${PERL_ROOT}/perl5" Function::Parameters Hash::Ordered PDL::CCS

cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
perl Makefile.PL INSTALL_BASE=${PERL_ROOT}/perl5
make install

cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
perl Makefile.PL INSTALL_BASE=${PERL_ROOT}/perl5
make install

cd ${MXNET_HOME}/perl-package/AI-MXNet/
perl Makefile.PL INSTALL_BASE=${PERL_ROOT}/perl5
make install

echo "Add to your shell startup script:"
echo "export LD_LIBRARY_PATH=${MXNET_HOME}/lib"
echo "export PERL5LIB=${PERL_ROOT}/perl5/lib/perl5"
