#!/usr/bin/env bash

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

# Build on Ubuntu 14.04 LTS for LINUX CPU/GPU
set -ex

# replace https with http to force apt-get update to use http
# nvidia-docker no longer supports ubuntu 14.04
# refer https://github.com/apache/incubator-mxnet/issues/18005
sudo sed -i 's/https/http/g' /etc/apt/sources.list.d/*.list
apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test -y
add-apt-repository ppa:openjdk-r/ppa -y # Java lib
apt-get update
apt-get install -y git \
    cmake3 \
    ninja-build \
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

curl -o apache-maven-3.3.9-bin.tar.gz -L http://www.eu.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz \
    || curl -o apache-maven-3.3.9-bin.tar.gz -L https://search.maven.org/remotecontent?filepath=org/apache/maven/apache-maven/3.3.9/apache-maven-3.3.9-bin.tar.gz

tar xzf apache-maven-3.3.9-bin.tar.gz
mkdir /usr/local/maven
mv apache-maven-3.3.9/ /usr/local/maven/
update-alternatives --install /usr/bin/mvn mvn /usr/local/maven/apache-maven-3.3.9/bin/mvn 1
update-ca-certificates -f

# patchelf available starting Ubuntu 16.04; compile from source for Ubuntu 14.04
mkdir /usr/local/patchelf
cd /usr/local/patchelf
curl -L -o patchelf-0.10.tar.gz https://github.com/NixOS/patchelf/archive/0.10.tar.gz
tar xzf patchelf-0.10.tar.gz
cd /usr/local/patchelf/patchelf-0.10
./bootstrap.sh
./configure
make
sudo make install
cd /

apt-get install -y python python-pip python3 python3-pip

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
# Restrict pip version to <19 due to use of Python 3.4 on Ubuntu 14.04
python3 -m pip install --upgrade 'pip<19'

# Restrict numpy version to <1.18 due to use of Python 3.4 on Ubuntu 14.04
python3 -m pip install --upgrade --ignore-installed nose cpplint==1.3.0 pylint==2.3.1 'numpy>1.16.0,<1.18' nose-timer 'requests<2.19.0,>=2.18.4' h5py==2.8.0rc1 scipy==1.0.1 boto3

# CMake 3.13.2+ is required
mkdir /opt/cmake && cd /opt/cmake
wget -nv https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.sh
sh cmake-3.13.5-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
rm cmake-3.13.5-Linux-x86_64.sh
cmake --version
