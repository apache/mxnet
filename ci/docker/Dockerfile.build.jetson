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

FROM mxnetci/dockcross-linux-arm64:05082018

ENV ARCH aarch64
ENV HOSTCC gcc
ENV TARGET ARMV8

# gh issue #11567 https://github.com/apache/incubator-mxnet/issues/11567
RUN sed -i '\#deb http://cdn-fastly.deb.debian.org/debian-security jessie/updates main#d' /etc/apt/sources.list
RUN sed -i 's/cdn-fastly.//' /etc/apt/sources.list


WORKDIR /work/deps

COPY install/ubuntu_arm.sh /work/
RUN /work/ubuntu_arm.sh

COPY install/arm_openblas.sh /work/
RUN /work/arm_openblas.sh

ENV OpenBLAS_HOME=${CROSS_ROOT}
ENV OpenBLAS_DIR=${CROSS_ROOT}

COPY install/deb_ubuntu_ccache.sh /work/
RUN /work/deb_ubuntu_ccache.sh

# Setup CUDA build env (including configuring and copying nvcc)
COPY --from=cudabuilder /usr/local/cuda /usr/local/cuda
ENV TARGET_ARCH aarch64
ENV TARGET_OS linux

# Install ARM depedencies based on Jetpack 3.3
RUN JETPACK_DOWNLOAD_PREFIX=https://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/3.3/lw.xd42/JetPackL4T_33_b39 && \
    CUDA_REPO_PREFIX=/var/cuda-repo-9-0-local && \
    ARM_CUDA_INSTALLER_PACKAGE=cuda-repo-l4t-9-0-local_9.0.252-1_arm64.deb && \
    ARM_CUDNN_INSTALLER_PACKAGE=libcudnn7_7.1.5.14-1+cuda9.0_arm64.deb && \
    ARM_CUDNN_DEV_INSTALLER_PACKAGE=libcudnn7-dev_7.1.5.14-1+cuda9.0_arm64.deb && \
    ARM_LICENSE_INSTALLER=cuda-license-9-0_9.0.252-1_arm64.deb && \
    ARM_CUBLAS_INSTALLER=cuda-cublas-9-0_9.0.252-1_arm64.deb && \
    ARM_NVINFER_INSTALLER_PACKAGE=libnvinfer4_4.1.3-1+cuda9.0_arm64.deb && \
    ARM_NVINFER_DEV_INSTALLER_PACKAGE=libnvinfer-dev_4.1.3-1+cuda9.0_arm64.deb && \
    dpkg --add-architecture arm64 && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_CUDA_INSTALLER_PACKAGE && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_CUDNN_INSTALLER_PACKAGE && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_CUDNN_DEV_INSTALLER_PACKAGE && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_NVINFER_INSTALLER_PACKAGE && \
    wget -nv $JETPACK_DOWNLOAD_PREFIX/$ARM_NVINFER_DEV_INSTALLER_PACKAGE && \
    dpkg -i --force-architecture  $ARM_CUDA_INSTALLER_PACKAGE && \
    apt-key add $CUDA_REPO_PREFIX/7fa2af80.pub && \
    dpkg -i --force-architecture  $ARM_CUDNN_INSTALLER_PACKAGE && \
    dpkg -i --force-architecture  $ARM_CUDNN_DEV_INSTALLER_PACKAGE && \
    dpkg -i --force-architecture  $CUDA_REPO_PREFIX/$ARM_LICENSE_INSTALLER && \
    dpkg -i --force-architecture  $CUDA_REPO_PREFIX/$ARM_CUBLAS_INSTALLER && \
    dpkg -i --force-architecture  $ARM_NVINFER_INSTALLER_PACKAGE && \
    dpkg -i --force-architecture  $ARM_NVINFER_DEV_INSTALLER_PACKAGE && \
    apt update -y || true && apt install -y cuda-libraries-dev-9-0 libcudnn7-dev libnvinfer-dev
ENV PATH $PATH:/usr/local/cuda/bin
ENV NVCCFLAGS "-m64"
ENV CUDA_ARCH "-gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62"
ENV NVCC /usr/local/cuda/bin/nvcc

COPY runtime_functions.sh /work/
WORKDIR /work/mxnet
