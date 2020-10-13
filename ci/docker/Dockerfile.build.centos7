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
        devtoolset-8 \
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
    yum clean all

# Make Python 3.6 and Maven 3.3 Software Collections available by default during
# the following build steps in this Dockerfile
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

# Fix the en_DK.UTF-8 locale to test locale invariance
RUN localedef -i en_DK -f UTF-8 en_DK.UTF-8

# Python dependencies
RUN python3 -m pip install --upgrade pip
COPY install/requirements /work/
RUN python3 -m pip install -r /work/requirements

ARG USER_ID=0
COPY install/docker_filepermissions.sh /work/
RUN /work/docker_filepermissions.sh

ENV PYTHONPATH=./python/
# Verify that MXNet works correctly when the C locale is set to a locale that uses a comma as the
# decimal separator. Please see #16134 for an example of a bug caused by incorrect handling of
# number serialization and deserialization.
ENV LC_NUMERIC=en_DK.UTF-8
WORKDIR /work/mxnet

COPY runtime_functions.sh /work/
