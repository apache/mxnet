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

# This script install pre-requisites required for building MXNet pypi package wheel.

sudo apt-get update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt install -y \
       build-essential \
       git \
       autoconf \
       libtool \
       unzip \
       gcc-4.8 \
       g++-4.8 \
       gfortran \
       gfortran-4.8 \
       nasm \
       make \
       automake \
       pkg-config \
       pandoc \
       python-dev \
       libssl-dev \
       python-pip

wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz
tar -xvzf cmake-3.12.3.tar.gz
cd cmake-3.12.3
./bootstrap
make -j
sudo make install

pip install -U pip "setuptools==36.2.0" wheel --user
pip install pypandoc numpy==1.15.0 --user
