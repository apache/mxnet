# -*- mode: dockerfile -*-
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
#
# Dockerfile to build libmxnet.so, and a python wheel for the Jetson TX1/TX2
# This script assumes /work/mxnet exists and contains the mxnet code you wish to compile and
# that /work/build exists and is the target for your output.

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV ARCH=aarch64 \
    HOSTCC=gcc \
    TARGET=ARMV8

WORKDIR /usr/local

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ninja-build \
    git \
    wget \
    zip \
    unzip \
    python3 \
    python3-pip \
    awscli \
    crossbuild-essential-arm64 \
 && rm -rf /var/lib/apt/lists/*

# cmake on Ubuntu 18.04 is too old
RUN python3 -m pip install cmake

# ccache on Ubuntu 18.04 is too old to support Cuda correctly
COPY install/deb_ubuntu_ccache.sh /work/
RUN /work/deb_ubuntu_ccache.sh

COPY toolchains/aarch64-linux-gnu-toolchain.cmake /usr
ENV CMAKE_TOOLCHAIN_FILE=/usr/aarch64-linux-gnu-toolchain.cmake

RUN git clone --recursive -b v0.3.12 https://github.com/xianyi/OpenBLAS.git && \
    cd /usr/local/OpenBLAS && \
    make NOFORTRAN=1 CC=aarch64-linux-gnu-gcc && \
    make PREFIX=/usr/aarch64-linux-gnu install && \
    cd /usr/local && \
    rm -rf OpenBLAS

# Install aarch64 cross depedencies based on Jetpack 4.4
# Dependencies require cuda-toolkit-10.2 which isn't installed in nvidia docker container
# It contains cuda-compat instead. However deb files currently depend on cuda-toolkit alone.
# Hence force dpkg configure
RUN wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cross-aarch64_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cudart-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cufft-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cupti-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-curand-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cusolver-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-cusparse-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-driver-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-misc-headers-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-npp-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-nsight-compute-addon-l4t-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-nvgraph-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-nvml-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cuda/cuda-nvrtc-cross-aarch64-10-2_10.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/c/cublas/libcublas-cross-aarch64_10.2.2.89-1_all.deb && \
    wget https://repo.download.nvidia.com/jetson/x86_64/pool/r32.4/n/nsight-compute/nsight-compute-addon-l4t-2019.5.0_2019.5.0.14-1_all.deb && \
    dpkg -i --force-all  *.deb && \
    rm *.deb && \
    apt-get update && \
    apt-get install -y -f && \
    apt-get install -y cuda-cross-aarch64 cuda-cross-aarch64-10-2 && \
    rm -rf /var/lib/apt/lists/*

# nvidia jetpack 4.4 installs libcublas.so at /usr/lib/aarch64-linux-gnu
# while previously it used to store it at /usr/local/cuda/targets/aarch64-linux/lib/stubs
RUN ln -s /usr/lib/aarch64-linux-gnu/libcublas.so /usr/local/cuda/targets/aarch64-linux/lib/stubs/libcublas.so

ARG USER_ID=0
ARG GROUP_ID=0
COPY install/ubuntu_adduser.sh /work/
RUN /work/ubuntu_adduser.sh

COPY runtime_functions.sh /work/
WORKDIR /work/mxnet
