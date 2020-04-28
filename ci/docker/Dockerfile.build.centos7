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
# Dockerfile for CentOS 7 based builds.
# Via the CentOS 7 Dockerfiles, we ensure MXNet continues to run fine on older systems.
#
# See docker-compose.yml for supported BASE_IMAGE ARGs and targets.

####################################################################################################
# The Dockerfile uses a dynamic BASE_IMAGE (for example centos:7,
# nvidia/cuda:10.2-cudnn7-devel-centos7 etc).
# On top of BASE_IMAGE we install all dependencies shared by all MXNet build
# environments into a "base" target. At the end of this file, we specialize
# "base" for specific usecases. The target built by docker can be selected via
# "--target" option or docker-compose.yml
####################################################################################################
ARG BASE_IMAGE
FROM $BASE_IMAGE AS base

WORKDIR /work/deps

RUN yum -y check-update || true && \
    yum -y install epel-release centos-release-scl && \
    yum install -y \
        # Utilities
        wget \
        unzip \
        patchelf \
        pandoc \
        # Development tools
        git \
        make \
        ninja-build \
        automake \
        autoconf \
        libtool \
        protobuf-compiler \
        protobuf-devel \
        # CentOS Software Collections https://www.softwarecollections.org
        devtoolset-7 \
        rh-python36 \
        rh-maven35 \
        # Libraries
        # Provide clbas headerfiles
        atlas-devel \
        openblas-devel \
        lapack-devel \
        opencv-devel \
        openssl-devel \
        zeromq-devel \
        # Build-dependencies for ccache 3.7.9
        gperf \
        libb2-devel \
        libzstd-devel && \
    yum clean all && \
    # Centos 7 only provides ninja-build
    ln -s /usr/bin/ninja-build /usr/bin/ninja

# Make GCC7, Python 3.5 and Maven 3.3 Software Collections available by default
# during build and runtime of this container
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7", "rh-python36", "rh-maven35" ]

# Install minimum required cmake version
RUN cd /usr/local/src && \
    wget -nv https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.sh && \
    sh cmake-3.13.5-Linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm cmake-3.13.5-Linux-x86_64.sh

# ccache 3.7.9 has fixes for caching nvcc outputs
RUN cd /usr/local/src && \
    git clone --recursive https://github.com/ccache/ccache.git && \
    cd ccache && \
    git checkout v3.7.9 && \
    ./autogen.sh && \
    ./configure --disable-man && \
    make -j$(nproc) && \
    make install && \
    cd /usr/local/src && \
    rm -rf ccache

# Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir pylint cython numpy requests h5py scipy==1.2.3 wheel \
    pytest==5.3.5 \
    pytest-env==0.6.2 \
    pytest-cov==2.8.1 \
    pytest-xdist==1.31.0 \
    pytest-timeout==1.3.4 \
    mock==2.0.0 \
    onnx==1.5.0 \
    protobuf==3.5.2 \
    tabulate==0.7.5

ARG USER_ID=0
COPY install/docker_filepermissions.sh /work/
RUN /work/docker_filepermissions.sh

ENV PYTHONPATH=./python/
WORKDIR /work/mxnet

COPY runtime_functions.sh /work/

####################################################################################################
# Specialize base image to install more gpu specific dependencies.
# The target built by docker can be selected via "--target" option or docker-compose.yml
####################################################################################################
FROM base as gpu
# NCCL is missing on CentOS7 images https://gitlab.com/nvidia/container-images/cuda/-/issues/68
# Install manually; different Cuda versions require different NCCL versions
# https://wiki.bash-hackers.org/syntax/pe#search_and_replace
RUN export SHORT_CUDA_VERSION=${CUDA_VERSION%.*} && \
    if [[ ${SHORT_CUDA_VERSION} == 9.2 ]]; then \
        export NCCL_VERSION=2.4.8; \
    elif [[ ${SHORT_CUDA_VERSION} == 10.* ]]; then \
        export NCCL_VERSION=2.6.4; \
    else \
        echo "ERROR: Cuda ${SHORT_CUDA_VERSION} not yet supported in Dockerfile.build.centos7"; \
        exit 1; \
    fi && \
    curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm -O && \
    rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm && \
    yum -y check-update || true && \
    yum -y install \
        libnccl-${NCCL_VERSION}-1+cuda${SHORT_CUDA_VERSION} \
        libnccl-devel-${NCCL_VERSION}-1+cuda${SHORT_CUDA_VERSION} \
        libnccl-static-${NCCL_VERSION}-1+cuda${SHORT_CUDA_VERSION} && \
    yum clean all
