<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

# Overview

This folder contains scripts for building the dependencies from source. The static libraries from
the build artifacts can be used to create self-contained shared object for mxnet through static
linking.

# Settings

The scripts use the following environment variables for setting behavior:

`DEPS_PATH`: the location in which the libraries are downloaded, built, and installed.
`PLATFORM`: name of the OS in lower case. Supported options are 'linux' and 'darwin'.

It also expects the following build tools in path: make, cmake, tar, unzip, autoconf, nasm

# FAQ

## Build failure regarding to gcc, g++, gfortran
Currently, we only support gcc-4.8 build. It's your own choice to use a higher version of gcc. Please make sure your gcc, g++ and gfortran always have the same version in order to eliminate build failure.

## idn2 not found
This issue appeared in the OSX build with XCode version 8.0 above (reproduced on 9.2). Please add the following build flag in `curl.sh` if your XCode version is more than 8.0:
```
--without-libidn2
``` 

***

# Dependency Update Runbook

MXNet is built on top of many dependencies. Managing these dependencies could be a big headache. This goal of this document is to give a overview of those dependencies and how to upgrade when new version of those are rolled out.

## Overview

The dependencies could be categorized by several groups: BLAS libraries, CPU-based performance boost library, i.e. oneDNN and GPU-based performance boosting library including CUDA, cuDNN, NCCL. and others including OpenCV, Numpy, S3-related, PS-lite dependencies. The list below shows all the dependencies and their version. Except for CUDA, cuDNN, NCCL which the user is required to install on their environments, we statically link those dependencies into libmxnet.so when we build PyPi package. By doing this, the user can take advantage of these dependencies without being worry about it.

| Dependencies  | MXNet Version |
| :------------: |:-------------:| 
|OpenBLAS| 0.3.9 |
|oneDNN| 2.6 |
|CUDA| 10.1 |
|cuDNN| 7.5.1 |
|NCCL| 2.4.2 |
|numpy| >1.16.0,<2.0.0 |
|request| >=2.20.0,< 3.0.0 |
|graphviz| <0.9.0,>=0.8.1 |
|OpenCV| 3.4.2 |
|zlib| 1.2.6 |
|libjpeg-turbo| 2.0.2 |
|libpng| 1.6.35 |
|libtiff| 4-0-10 |
|eigen| 3.3.4 |
|libcurl| 7.61.0 |
|libssl-dev| 1.1.1b |
|zmq| 4.2.2 |
|protobuf| 3.5.1 |
|lz4| r130 |
|cityhash| 1.1.1 |

## How to update them?

#### 0. Prerequisite Software
```
sudo apt update
sudo apt-get install -y git \
    cmake \
    libcurl4-openssl-dev \
    unzip \
    gcc-4.8 \
    g++-4.8 \
    gfortran \
    gfortran-4.8 \
    binutils \
    nasm \
    libtool \
    curl \
    wget \
    sudo \
    gnupg \
    gnupg2 \
    gnupg-agent \
    pandoc \
    python3-pip \
    automake \
    pkg-config
```

### MKL, oneDNN

@pengzhao-intel (https://github.com/apache/mxnet/commits?author=pengzhao-intel) and his team are tracking and updating these versions. Kudos to them!

### CUDA, cuDNN, NCCL

#### 1. Environment Setup
We will install all the prerequsite software.
We demonstrate with CUDA10/cuDNN7.5/NCCL 2.4.2.
You might want to change these versions to suit your needs.

```    
# CUDA installation 
# Take CUDA 10 for example, please follow the instructions on https://developer.nvidia.com/cuda-downloads
# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 410.48?
# (y)es/(n)o/(q)uit: y
# 
# Do you want to install the OpenGL libraries?
# (y)es/(n)o/(q)uit [ default is yes ]:
#
# Do you want to run nvidia-xconfig?
# This will update the system X configuration file so that the NVIDIA X driver
# is used. The pre-existing X configuration file will be backed up.
# This option should not be used on systems that require a custom
# X configuration, such as systems with multiple GPU vendors.
# (y)es/(n)o/(q)uit [ default is no ]:
# 
# Install the CUDA 10.0 Toolkit?
# (y)es/(n)o/(q)uit: y
#
# Enter Toolkit Location
# [ default is /usr/local/cuda-10.0 ]:
#
# Do you want to install a symbolic link at /usr/local/cuda?
# (y)es/(n)o/(q)uit: y
#
# Install the CUDA 10.0 Samples?
# (y)es/(n)o/(q)uit: n

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Check installation
nvidia-smi

# cuDNN Setup 
# Take cuDNN 7.5.0 with CUDA 10 for example
# https://developer.nvidia.com/rdp/cudnn-download
# Register with NVIDIA and download cudnn-10.0-linux-x64-v7.5.0.56.tgz
# scp it to your instance
# https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
tar -xvzf cudnn-10.0-linux-x64-v7.5.0.56.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# Check cuDNN version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 
# #define CUDNN_MAJOR 7
# #define CUDNN_MINOR 5
# #define CUDNN_PATCHLEVEL 0
# --
# #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
#
# #include "driver_types.h"

# install NCCL
# take NCCL 2.4.2 for example
# https://developer.nvidia.com/nccl/nccl2-download-survey
# Register with NVIDIA and download nccl-repo-ubuntu1604-2.4.2-ga-cuda10.0_1-1_amd64.deb
sudo dpkg -i nccl-repo-ubuntu1604-2.4.2-ga-cuda10.0_1-1_amd64.deb
sudo apt-key add /var/nccl-repo-2.4.2-ga-cuda10.0/7fa2af80.pub
sudo apt update
sudo apt install libnccl2 libnccl-dev
# we will check the NCCL version later
```
#### 2. Build
We will build MXNet with statically linked dependencies.
```
# Clone MXNet repo
git clone --recursive https://github.com/apache/mxnet.git
cd mxnet
# Make sure you pin to specific commit for all the performance sanity check to make fair comparison
# Make corresponding change on tools/setup_gpu_build_tools.sh
# to upgrade CUDA version, please refer to PR #14887.
# Make sure you add new makefile and right debs CUDA uses on the website
# http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/

# Build PyPi package
tools/staticbuild/build.sh cu100mkl

# Wait for 10 - 30 mins, you will find libmxnet.so under the mxnet/lib

# Install python frontend
pip install -e python
# Test MXNet
>>> import mxnet as mx
>>> mx.runtime.feature_list()

# Test NCCL version
export NCCL_DEBUG=VERSION
vim tests/python/gpu/test_nccl.py
# Remove @unittest.skip("Test requires NCCL library installed and enabled during build") then run
pytest --verbose tests/python/gpu/test_nccl.py
# test_nccl.test_nccl_pushpull ... NCCL version 2.4.2+cuda10.0
# ok
# ----------------------------------------------------------------------
# Ran 1 test in 67.666s

OK
```
#### 3. Performance Sanity Check
Please run performance test aginast the MXNet you build before raising the PR.

#### 4. Raise a PR
1. Update the tools/setup_gpu_build_tools.sh please refer to PR [#14988](https://github.com/apache/mxnet/pull/14988), [#14887](https://github.com/apache/mxnet/pull/14887/files)
2. (optional) Update the CI-related configuration/shell script/Dockerfile. Please refer to PR [#14986](https://github.com/apache/mxnet/pull/14986/files), [#14950](https://github.com/apache/mxnet/pull/14950/files)

#### 5. CI Test
1. Our CI would test PyPi and Scala publish of latest CUDA version i.e. mxnet-cu101mkl

### numpy, requests, graphviz (python dependencies)
1. Please refer to [#14588](https://github.com/apache/mxnet/pull/14588/files) and make sure the version have both of upper bound and lower bound
#### Checklist
- [ ] Python/setup.py
- [ ] tools/pip/setup.py
- [ ] ci/docker/install/requirements
- [ ] ci/docker/install/ubuntu_python.sh
- [ ] ci/qemu/mxnet_requirements.txt
- [ ] docs/install/requirements.txt 

2. Build from source to do sanity check
```
# Compile mxnet to get libmxnet.so
pip install -e python
python
>>> import mxnet as mx
>>> mx.nd.ones((1, 2))
[[1. 1.]]
<NDArray 1x2 @cpu(0)>
```

### OpenCV and its dependencies: zlib, libjpeg-turbo, libpng, libtiff, eigen

#### Update the build script
1. Find the library under `tools/dependencies` and update the version.

#### Sanity Check
1. Environment Setup
```python
# Take Ubuntu 16.04 for example
sudo apt update
sudo apt-get install -y git \
    cmake \
    libcurl4-openssl-dev \
    unzip \
    gcc-4.8 \
    g++-4.8 \
    gfortran \
    gfortran-4.8 \
    binutils \
    nasm \
    libtool \
    curl \
    wget \
    sudo \
    gnupg \
    gnupg2 \
    gnupg-agent \
    pandoc \
    python3-pip \
    automake \
    pkg-config
```
2. Build PyPi package
```
# Update the dependency under tools/dependencies, then
tools/staticbuild/build.sh mkl

# Wait for 10 - 30 mins, you will find libmxnet.so under the mxnet/lib

# Install python frontend
pip install -e python
# Test MXNet
>>> import mxnet as mx
>>> mx.runtime.feature_list()
```

3. Run performance tests against image related tasks


### Other dependencies under tools/dependencies

#### Update the build script
1. Find the library under `tools/dependencies` and update the version.

#### Sanity Check
1. Environment Setup
```python
# Take Ubuntu 16.04 for example
sudo apt update
sudo apt-get install -y git \
    cmake \
    libcurl4-openssl-dev \
    unzip \
    gcc-4.8 \
    g++-4.8 \
    gfortran \
    gfortran-4.8 \
    binutils \
    nasm \
    libtool \
    curl \
    wget \
    sudo \
    gnupg \
    gnupg2 \
    gnupg-agent \
    pandoc \
    python3-pip \
    automake \
    pkg-config
```
2. Build PyPi package
```
# Update the dependency under tools/dependencies, then
tools/staticbuild/build.sh mkl

# Wait for 10 - 30 mins, you will find libmxnet.so under the mxnet/lib

# Install python frontend
pip install -e python
# Test MXNet
>>> import mxnet as mx
>>> mx.runtime.feature_list()
```
