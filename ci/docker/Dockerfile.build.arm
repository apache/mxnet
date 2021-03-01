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
# Dockerfile to build MXNet for ARM

####################################################################################################
# Shared base for all ARM targets
####################################################################################################
FROM ubuntu:20.04 AS base

WORKDIR /usr/local

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ninja-build \
    cmake \
    ccache \
    git \
    curl \
    zip \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

ARG USER_ID=0
ARG GROUP_ID=0
COPY install/ubuntu_adduser.sh /work/
RUN /work/ubuntu_adduser.sh

COPY runtime_functions.sh /work/


####################################################################################################
# Specialize base image for ARMv6
####################################################################################################
FROM base as armv6

ENV ARCH=armv6l \
    HOSTCC=gcc \
    HOSTCXX=g++ \
    TARGET=ARMV6

# We use a toolchain from toolchains.bootlin.com instead of Debian / Ubunut
# crossbuild-essential-armel toolchain, as the latter targets ARM architecture
# versions 4T, 5T, and 6, whereas we only wish to target ARMV6 and like to use
# ARMV6 specific features. https://wiki.debian.org/ArmEabiPort
RUN curl -o armv6-eabihf--glibc--stable-2020.02-2.tar.bz2 -L https://toolchains.bootlin.com/downloads/releases/toolchains/armv6-eabihf/tarballs/armv6-eabihf--glibc--stable-2020.02-2.tar.bz2 && \
    tar xf armv6-eabihf--glibc--stable-2020.02-2.tar.bz2 && \
    rm armv6-eabihf--glibc--stable-2020.02-2.tar.bz2
ENV CMAKE_TOOLCHAIN_FILE=/usr/local/armv6-eabihf--glibc--stable-2020.02-2/share/buildroot/toolchainfile.cmake

RUN git clone --recursive -b v0.3.12 https://github.com/xianyi/OpenBLAS.git && \
    cd /usr/local/OpenBLAS && \
    make NOFORTRAN=1 NO_SHARED=1 CC=/usr/local/armv6-eabihf--glibc--stable-2020.02-2/bin/arm-linux-gcc && \
    make PREFIX=/usr/local/armv6-eabihf--glibc--stable-2020.02-2/arm-buildroot-linux-gnueabihf/sysroot NO_SHARED=1 install && \
    cd /usr/local && \
    rm -rf OpenBLAS

WORKDIR /work/mxnet


####################################################################################################
# Specialize base image for ARMv7
####################################################################################################
FROM base as armv7

ENV ARCH=armv7l \
    HOSTCC=gcc \
    HOSTCXX=g++ \
    TARGET=ARMV7

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    crossbuild-essential-armhf \
 && rm -rf /var/lib/apt/lists/*

COPY toolchains/arm-linux-gnueabihf-toolchain.cmake /usr/local
ENV CMAKE_TOOLCHAIN_FILE=/usr/local/arm-linux-gnueabihf-toolchain.cmake

RUN git clone --recursive -b v0.3.12 https://github.com/xianyi/OpenBLAS.git && \
    cd /usr/local/OpenBLAS && \
    make NOFORTRAN=1 NO_SHARED=1 CC=arm-linux-gnueabihf-gcc && \
    make PREFIX=/usr/local/arm-linux-gnueabihf NO_SHARED=1 install && \
    cd /usr/local && \
    rm -rf OpenBLAS

RUN git clone --recursive -b v1.2.11 https://github.com/madler/zlib.git && \
    cd /usr/local/zlib && \
    CHOST=arm \
    CC=arm-linux-gnueabihf-gcc \
    AR=arm-linux-gnueabihf-ar \
    RANLIB=arm-linux-gnueabihf-ranlib \
    ./configure --static --prefix=/usr/local/arm-linux-gnueabihf && \
    make -j$(nproc) && \
    make install && \
    cd /usr/local && \
    rm -rf zlib

WORKDIR /work/mxnet


####################################################################################################
# Specialize base image for ARMv8
####################################################################################################
FROM base as armv8

ENV ARCH=aarch64 \
    HOSTCC=gcc \
    HOSTCXX=g++ \
    TARGET=ARMV8

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    crossbuild-essential-arm64 \
 && rm -rf /var/lib/apt/lists/*

COPY toolchains/aarch64-linux-gnu-toolchain.cmake /usr
ENV CMAKE_TOOLCHAIN_FILE=/usr/aarch64-linux-gnu-toolchain.cmake

RUN git clone --recursive -b v0.3.12 https://github.com/xianyi/OpenBLAS.git && \
    cd /usr/local/OpenBLAS && \
    make NOFORTRAN=1 NO_SHARED=1 CC=aarch64-linux-gnu-gcc && \
    make PREFIX=/usr/aarch64-linux-gnu NO_SHARED=1 install && \
    cd /usr/local && \
    rm -rf OpenBLAS

RUN git clone --recursive -b v1.2.11 https://github.com/madler/zlib.git && \
    cd /usr/local/zlib && \
    CHOST=arm \
    CC=aarch64-linux-gnu-gcc \
    AR=aarch64-linux-gnu-ar \
    RANLIB=aarch64-linux-gnu-ranlib \
    ./configure --static --prefix=/usr/aarch64-linux-gnu && \
    make -j$(nproc) && \
    make install && \
    cd /usr/local && \
    rm -rf zlib

WORKDIR /work/mxnet
