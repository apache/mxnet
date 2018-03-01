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
set -ex

# This script handles installation of build and test dependencies
# for different platforms.

#############################
# Ubuntu Dependencies:

ubuntu_install_all_deps() {
    set -ex
    ubuntu_install_core
    ubuntu_install_python
    ubuntu_install_scala
    ubuntu_install_r
    ubuntu_install_perl
    ubuntu_install_lint
}


ubuntu_install_core() {
    set -ex
    apt-get update
    apt-get install -y \
        build-essential \
        git \
        libopenblas-dev \
        liblapack-dev \
        libopencv-dev \
        libcurl4-openssl-dev \
        cmake \
        wget \
        unzip \
        sudo \
        ninja-build \
        python-pip

    # Link Openblas to Cblas as this link does not exist on ubuntu16.04
    ln -s /usr/lib/libopenblas.so /usr/lib/libcblas.so
    pip install cpplint==1.3.0 pylint==1.8.2
}

ubuntu_install_nvidia() {
    set -ex
    apt install -y software-properties-common
    add-apt-repository -y ppa:graphics-drivers
    # Retrieve ppa:graphics-drivers and install nvidia-drivers.
    # Note: DEBIAN_FRONTEND required to skip the interactive setup steps
    apt update
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends cuda-8-0
}

ubuntu_install_perl() {
    set -ex
    # install libraries for mxnet's perl package on ubuntu
    apt-get install -y libmouse-perl pdl cpanminus swig libgraphviz-perl
    cpanm -q Function::Parameters Hash::Ordered
}

ubuntu_install_python() {
    set -ex
    # install libraries for mxnet's python package on ubuntu
    apt-get install -y python-dev python3-dev virtualenv

    # the version of the pip shipped with ubuntu may be too lower, install a recent version here
    wget -nv https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
    python2 get-pip.py

    pip2 install nose pylint numpy nose-timer requests h5py scipy
    pip3 install nose pylint numpy nose-timer requests h5py scipy
}

ubuntu_install_r() {
    set -ex
    # install libraries for mxnet's r package on ubuntu
    echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list
    gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
    gpg -a --export E084DAB9 | apt-key add -

    apt-get update
    apt-get install -y r-base r-base-dev libxml2-dev libssl-dev libxt-dev
}

ubuntu_install_scala() {
    set -ex
    # install libraries for mxnet's scala package on ubuntu
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:webupd8team/java
    apt-get update
    echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
    apt-get install -y oracle-java8-installer
    apt-get install -y oracle-java8-set-default
    apt-get update && apt-get install -y maven
}


ubuntu_install_lint() {
    set -ex
    apt-get update
    apt-get install -y python-pip sudo
    #pip install cpplint==1.3.0 pylint==1.8.2
}

#############################
# Amazon Linux

amzn_linux_install_all_deps() {
    pushd .
    mkdir deps
    amzn_linux_install_core
    amzn_linux_install_opencv
    amzn_linux_install_openblas
    amzn_linux_install_python2
    amzn_linux_install_python3
    amzn_linux_install_testdeps
    amzn_linux_install_julia
    amzn_linux_install_maven
    amzn_linux_install_library
    popd
}

amzn_linux_install_core() {
    set -ex
    pushd .
    yum install -y git
    yum install -y wget
    yum install -y sudo
    yum install -y re2c
    yum groupinstall -y 'Development Tools'

    # Ninja
    git clone --recursive https://github.com/ninja-build/ninja.git
    cd ninja
    ./configure.py --bootstrap
    cp ninja /usr/local/bin
    popd

    # CMake
    pushd .
    git clone --recursive https://github.com/Kitware/CMake.git --branch v3.10.2
    cd CMake
    ./bootstrap
    make -j$(nproc)
    make install
    popd
}

amzn_linux_install_opencv() {
    set -ex
    pushd .
    yum install -y python27 python27-setuptools
    git clone https://github.com/opencv/opencv
    cd opencv
    mkdir -p build
    cd build
    cmake -DBUILD_opencv_gpu=OFF -DWITH_EIGEN=ON -DWITH_TBB=ON -DWITH_CUDA=OFF -DWITH_1394=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -GNinja ..
    #make PREFIX=/usr/local install
    ninja install
    popd
}

amzn_linux_install_openblas() {
    set -ex
    pushd .
    git clone https://github.com/xianyi/OpenBLAS
    cd OpenBLAS
    make FC=gfortran -j $(($(nproc) + 1))
    make PREFIX=/usr/local install
    popd
}

amzn_linux_install_python2() {
    set -ex
    yum groupinstall -y "Development Tools"
    yum install -y mlocate python27 python27-setuptools python27-tools python27-numpy python27-scipy python27-nose python27-matplotlib unzip
    ln -s -f /usr/bin/python2.7 /usr/bin/python2
    wget -nv https://bootstrap.pypa.io/get-pip.py
    python2 get-pip.py
    $(which easy_install-2.7) --upgrade pip
    if [ -f /usr/local/bin/pip ] && [ -f /usr/bin/pip ]; then
      mv /usr/bin/pip /usr/bin/pip.bak
      ln /usr/local/bin/pip /usr/bin/pip
    fi

    ln -s -f /usr/local/bin/pip /usr/bin/pip
    for i in ipython[all] jupyter pandas scikit-image h5py pandas sklearn sympy; do echo "${i}..."; pip install -U $i >/dev/null; done
}


amzn_linux_install_python3() {
    set -ex
    pushd .
    wget -nv https://bootstrap.pypa.io/get-pip.py
    mkdir py3
    cd py3
    wget -nv https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
    tar -xvzf Python-3.5.2.tgz
    cd Python-3.5.2
    yum install -y zlib-devel openssl-devel sqlite-devel bzip2-devel gdbm-devel ncurses-devel xz-devel readline-devel
    ./configure --prefix=/opt/ --with-zlib-dir=/usr/lib64
    make -j$(nproc)
    mkdir /opt/bin
    mkdir /opt/lib
    make install
    ln -s -f /opt/bin/python3 /usr/bin/python3
    cd ../..
    python3 get-pip.py
    ln -s -f /opt/bin/pip /usr/bin/pip3

    mkdir -p /home/jenkins/.local/lib/python3.5/site-packages/
    pip3 install numpy
    popd
}

amzn_linux_install_testdeps() {
    set -ex
    pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6'
    pip3 install nose
    ln -s -f /opt/bin/nosetests /usr/local/bin/nosetests3
    ln -s -f /opt/bin/nosetests-3.4 /usr/local/bin/nosetests-3.4
}

amzn_linux_install_julia() {
    set -ex
    wget -nv https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.0-linux-x86_64.tar.gz
    mv julia-0.5.0-linux-x86_64.tar.gz /tmp/
    tar xfvz /tmp/julia-0.5.0-linux-x86_64.tar.gz
    rm -f /tmp/julia-0.5.0-linux-x86_64.tar.gz
    # tar extracted in current directory
    ln -s -f ${PWD}/julia-3c9d75391c/bin/julia /usr/bin/julia
}

amzn_linux_install_maven() {
    set -ex
    wget -nv http://mirrors.ocf.berkeley.edu/apache/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
    mv apache-maven-3.3.9-bin.tar.gz /tmp/
    tar xfvz /tmp/apache-maven-3.3.9-bin.tar.gz
    yum install -y java-1.8.0-openjdk-devel
}


amzn_linux_install_library() {
    yum -y install graphviz
    pip install graphviz
    pip install opencv-python
}

centos7_all_deps() {
    set -ex
    centos7_install_core
    centos7_install_python
}

centos7_install_core() {
    # Multipackage installation does not fail in yum
    yum -y install epel-release
    yum -y install git
    yum -y install wget
    yum -y install atlas-devel # Provide clbas headerfiles
    yum -y install openblas-devel
    yum -y install lapack-devel
    yum -y install opencv-devel
    yum -y install openssl-devel
    yum -y install gcc-c++
    yum -y install make
    yum -y install cmake
    yum -y install wget
    yum -y install unzip
    yum -y install ninja-build
}

centos7_install_python() {
    # Python 2.7 is installed by default, install 3.6 on top
    yum -y install https://centos7.iuscommunity.org/ius-release.rpm
    yum -y install python36u

    # Install PIP
    curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
    python2.7 get-pip.py
    python3.6 get-pip.py

    pip2 install nose pylint numpy nose-timer requests h5py scipy
    pip3 install nose pylint numpy nose-timer requests h5py scipy
}

install_mkml() {
    set -ex
    pushd .
    wget -nv --no-check-certificate -O /tmp/mklml.tgz https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
    tar -zxvf /tmp/mklml.tgz && cp -rf mklml_*/* /usr/local/ && rm -rf mklml_*
    # ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/lib/gcc/x86_64-linux-gnu/5/
    popd
}


arm64_install_all_deps() {
    arm64_install_openblas
}

arm64_install_openblas() {
    set -ex
    pushd .
    wget -nv https://api.github.com/repos/xianyi/OpenBLAS/git/refs/heads/master -O openblas_version.json
    echo "Using openblas:"
    cat openblas_version.json
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    make -j$(nproc) TARGET=ARMV8
    make install
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
    ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/libopenblas.a
    ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/liblapack.a
    popd
}

android_arm64_install_all_deps() {
    android_arm64_install_ndk
    android_arm64_install_openblas
}

android_arm64_install_openblas() {
    set -ex
    pushd .
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    make -j$(nproc) TARGET=ARMV8 ARM_SOFTFP_ABI=1 HOSTCC=gcc NOFORTRAN=1 libs
    cp libopenblas.a /usr/local/lib
    popd
}

android_arm64_install_ndk() {
    set -ex
    pushd .
    export ANDROID_NDK_REVISION=15c
    curl -O https://dl.google.com/android/repository/android-ndk-r${ANDROID_NDK_REVISION}-linux-x86_64.zip && \
    unzip ./android-ndk-r${ANDROID_NDK_REVISION}-linux-x86_64.zip && \
    cd android-ndk-r${ANDROID_NDK_REVISION} && \
    ./build/tools/make_standalone_toolchain.py \
      --stl=libc++ \
      --arch arm64 \
      --api 21 \
      --install-dir=${CROSS_ROOT} && \

    popd
}

##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

$0: Execute a function by passing it as an argument to the script:

Possible commands:

EOF
    declare -F | cut -d' ' -f3
    echo
fi
