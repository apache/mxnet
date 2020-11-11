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
# Dockerfile to build MXNet for Android

####################################################################################################
# Shared base for all Android targets
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
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN curl -o android-ndk-r19c-linux-x86_64.zip -L https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip && \
    unzip android-ndk-r19c-linux-x86_64.zip && \
    rm android-ndk-r19c-linux-x86_64.zip
ENV CMAKE_TOOLCHAIN_FILE=/usr/local/android-ndk-r19c/build/cmake/android.toolchain.cmake

ARG USER_ID=0
ARG GROUP_ID=0
COPY install/ubuntu_adduser.sh /work/
RUN /work/ubuntu_adduser.sh

COPY runtime_functions.sh /work/


####################################################################################################
# Specialize base image for ARMv7
####################################################################################################
FROM base as armv7
ENV ARCH=armv7l \
    HOSTCC=gcc \
    HOSTCXX=g++ \
    TARGET=ARMV7

RUN git clone --recursive -b v0.3.12 https://github.com/xianyi/OpenBLAS.git && \
    cd /usr/local/OpenBLAS && \
    export TOOLCHAIN=/usr/local/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64 && \
    make NOFORTRAN=1 ARM_SOFTFP_ABI=1 NO_SHARED=1 \
        LDFLAGS="-L/usr/local/android-ndk-r19c/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/lib/gcc/arm-linux-androideabi/4.9.x -lm" \
        CC=$TOOLCHAIN/bin/armv7a-linux-androideabi16-clang AR=$TOOLCHAIN/bin/arm-linux-androideabi-ar && \
    make PREFIX=/usr/local/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/ NO_SHARED=1 install && \
    cd /usr/local && \
    rm -rf OpenBLAS

WORKDIR /work/build


####################################################################################################
# Specialize base image for ARMv8
####################################################################################################
FROM base as armv8
ENV ARCH=aarch64 \
    HOSTCC=gcc \
    HOSTCXX=g++ \
    TARGET=ARMV8

RUN git clone --recursive -b v0.3.12 https://github.com/xianyi/OpenBLAS.git && \
    cd /usr/local/OpenBLAS && \
    export TOOLCHAIN=/usr/local/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64 && \
    make NOFORTRAN=1 NO_SHARED=1 \
        LDFLAGS="-L/usr/local/android-ndk-r21/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/lib/gcc/aarch64-linux-android/4.9.x -lm" \
        CC=$TOOLCHAIN/bin/aarch64-linux-android21-clang AR=$TOOLCHAIN/bin/aarch64-linux-android-ar && \
    make PREFIX=/usr/local/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/ NO_SHARED=1 install && \
    cd /usr/local && \
    rm -rf OpenBLAS

WORKDIR /work/build
