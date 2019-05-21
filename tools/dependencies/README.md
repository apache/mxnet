<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

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

MXNet is built on top of many dependencies. Managing those dependencies could be a big headache. This goal of this document is to give a overview of those dependencies and how to upgrade when new version of those are rolled out.

## Overview

The dependencies could be categorized by several groups: BLAS libraries, CPU-based performance boost library i.e. MKLDNN and GPU-based performance boost library including CUDA, cuDNN, NCCL. and others including OpenCV, Numpy, S3-related, PS-lite dependencies. The list below shows all the dependencies and their version. Except for CUDA, cuDNN, NCCL, we statically link those dependencies into libmxnet.so when we build PyPi package. The user doesn't need to worry about it.


| Dependencies  | MXNet Version |
| :------------: |:-------------:| 
|MKL| N/A | 
|MKLDNN| 0.19      | 
|CUDA| 10.1      |
|cuDNN| 7.5.1     |
|NCCL| 2.4.2     |
|numpy| >1.16.0,<2.0.0 |
|request| >=2.20.0,< 3.0.0 |
|graphviz| <0.9.0,>=0.8.1 |
|OpenCV|3.4.2|
|zlib|1.2.6|
|libjpeg-turbo|1.5.90|
|libpng|1.6.35|
|libjpeg-turbo|2.0.2|
|libtiff|4-0-10|
|eigen|3.3.4|
|libcurl|7.61.0|
|libssl-dev|1.0.2l|
|zmq|4.2.2|
|protobuf|3.5.1|
|lz4|r130|
|cityhash|1.1.1|
|openssl|1.1.b|

# MKL, MKLDNN

@pengzhao-intel (https://github.com/apache/incubator-mxnet/commits?author=pengzhao-intel) and his team are tracking and updating these versions.

# CUDA, cuDNN, NCCL
#### 1. Environment Setup
```
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
    pkg-config \
    openjdk-8-jdk
    
# CUDA installation 
# take CUDA 10 for example
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod +x cuda_10.0.130_410.48_linux && sudo ./cuda_10.0.130_410.48_linux
# Installation except:
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
# take cuDNN 7.5.0 with CUDA 10 for example
# https://developer.nvidia.com/rdp/cudnn-download
# Register with NVIDIA and download cudnn-10.0-linux-x64-v7.5.0.56.tgz
# scp it to your instance
# https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
tar -xvzf cudnn-10.0-linux-x64-v7.5.0.56.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# check cuDNN version
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
# we will check the nccl version later
sudo apt install libnccl2 libnccl-dev
```
#### 2. Build 
```
# clone MXNet repo
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
# test build PyPi package
tools/staticbuild/build.sh cu100mkl pip

# wait for 10 - 30 mins, you will find libmxnet.so under the incubator-mxnet/lib

# install python frontend
cd python
pip3 install -e . --pre
# test MXNet
>>> import mxnet as mx
>>> mx.nd.ones((2, 5) ctx=mx.gpu(0))
```
#### 3. Performance Sanity Check
We will test 3 basic models
###### ResNet50 with ImageNet
```
# please configure aws client before this
aws s3 sync s3://aws-ml-platform-datasets/imagenet/pass-through/ ~/data/
# install prerequisite package
pip2 install psutil --user
pip2 install pandas --upgrade --user
pip install gluoncv==0.2.0b20180625 --user
# clone the testing script
git clone https://github.com/rahul003/deep-learning-benchmark-mirror.git*
# command 
python mxnet_benchmark/train_imagenet.py --use-rec --batch-size 128 --dtype float32 —num-data-workers 40 —num-epochs 3 —gpus 0 --lr 0.05 —warmup-epochs 5 --last-gamma —mode symbolic —model resnet50_v1b —rec-train /home/ubuntu/data/train-passthrough.rec —rec-train-idx /home/ubuntu/data/train-passthrough.idx —rec-val /home/ubuntu/data/val-passthrough.rec —rec-val-idx /home/ubuntu/data/val-passthrough.idx
# if you want to run above command multiple times, remember to delete log file
rm metrics_parameters_images_top_1.log
```
The throughput should be around `2800`
###### LSTM training with PTB
```
# make sure you install prerequisite package: psutil, pandas
# download testing script
git clone https://github.com/awslabs/deeplearning-benchmark.git
# command
python2 benchmark_driver.py --framework mxnet --task-name mkl_lstm_ptb_symbolic —num-gpus 1 --epochs 10 --metrics-suffix test --kvstore local
# if you want to run above command twice, remember to delete log file
rm mkl_lstm_ptb_symbolic.log
```
The throughput should be around `1000`
###### MLP with MNIST
```
# make sure you install prerequisite package: psutil, pandas
# download testing script
git clone https://github.com/awslabs/deeplearning-benchmark.git
```
please copy the put the following script to deeplearning-benchmark/mlp.py
@TODO
```python

```


