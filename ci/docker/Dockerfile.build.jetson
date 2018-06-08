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

FROM nvidia/cuda:9.0-cudnn7-devel as cudabuilder

# Temporary fix due to https://github.com/apache/incubator-mxnet/issues/10837
# FROM dockcross/linux-arm64
FROM mxnetci/dockcross-linux-arm64:05082018

ENV ARCH aarch64
ENV FC /usr/bin/${CROSS_TRIPLE}-gfortran
ENV HOSTCC gcc
ENV TARGET ARMV8

WORKDIR /work

# Build OpenBLAS
RUN git clone --recursive -b v0.2.20 https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    make -j$(nproc) && \
    PREFIX=${CROSS_ROOT} make install

# Setup CUDA build env (including configuring and copying nvcc)
COPY --from=cudabuilder /usr/local/cuda /usr/local/cuda
ENV TARGET_ARCH aarch64
ENV TARGET_OS linux

# Install ARM depedencies based on Jetpack 3.2
RUN JETPACK_DOWNLOAD_PREFIX=http://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/3.2GA/m892ki/JetPackL4T_32_b196/ && \
    ARM_CUDA_INSTALLER_PACKAGE=cuda-repo-l4t-9-0-local_9.0.252-1_arm64.deb && \
    ARM_CUDNN_INSTALLER_PACKAGE=libcudnn7_7.0.5.13-1+cuda9.0_arm64.deb && \
    ARM_CUDNN_DEV_INSTALLER_PACKAGE=libcudnn7-dev_7.0.5.13-1+cuda9.0_arm64.deb && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_CUDA_INSTALLER_PACKAGE && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_CUDNN_INSTALLER_PACKAGE && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_CUDNN_DEV_INSTALLER_PACKAGE && \
    dpkg -i $ARM_CUDA_INSTALLER_PACKAGE && \
    apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub && \
    dpkg -i $ARM_CUDNN_INSTALLER_PACKAGE && \
    dpkg -i $ARM_CUDNN_DEV_INSTALLER_PACKAGE && \
    apt update -y && apt install -y unzip cuda-libraries-dev-9-0 libcudnn7-dev

ENV PATH $PATH:/usr/local/cuda/bin
ENV NVCCFLAGS "-m64"
ENV CUDA_ARCH "-gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62"
ENV NVCC /usr/local/cuda/bin/nvcc

COPY runtime_functions.sh /work/
WORKDIR /work/mxnet
