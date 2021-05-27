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

# Build on Ubuntu 18.04 LTS for LINUX CPU/GPU
set -ex

apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test -y
add-apt-repository ppa:openjdk-r/ppa -y # Java lib
apt-get update
apt-get install -y git \
    build-essential \
    ninja-build \
    libssl-dev \
    libcurl4-openssl-dev \
    ccache \
    unzip \
    libtool \
    curl \
    wget \
    sudo \
    gnupg \
    gnupg2 \
    gnupg-agent \
    gcc \
    g++ \
    gfortran \
    libc6-lse \
    pandoc \
    python3 \
    python3-pip \
    automake \
    pkg-config \
    openjdk-8-jdk \
    patchelf

# build gcc-8.5 from source
apt update
apt install -y flex bison
wget https://ftpmirror.gnu.org/gcc/gcc-8.5.0/gcc-8.5.0.tar.xz
tar xf gcc-8.5.0.tar.xz
cd gcc-8.5.0/
sed -i contrib/download_prerequisites -e '/base_url=/s/ftp/http/'
contrib/download_prerequisites
cd ..
mkdir build && cd build
../gcc-8.5.0/configure -v --build=aarch64-linux-gnu --host=aarch64-linux-gnu --target=aarch64-linux-gnu --prefix=/usr/local/gcc-8.5.0 --enable-checking=release --enable-languages=c,c++,fortran --disable-multilib --program-suffix=-8.5
make -j$(nproc)
sudo make install-strip
cd ..
export export PATH=/usr/local/gcc-8.5.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/gcc-8.5.0/lib64:$LD_LIBRARY_PATH
update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-8.5.0/bin/gcc-8.5 100 --slave /usr/bin/g++ g++ /usr/local/gcc-8.5.0/bin/g++-8.5 --slave /usr/bin/gcov gcov /usr/local/gcc-8.5.0/bin/gcov-8.5 --slave /usr/bin/gfortran gfortran /usr/local/gcc-8.5.0/bin/gfortran-8.5

curl -o apache-maven-3.3.9-bin.tar.gz -L http://www.eu.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz \
    || curl -o apache-maven-3.3.9-bin.tar.gz -L https://search.maven.org/remotecontent?filepath=org/apache/maven/apache-maven/3.3.9/apache-maven-3.3.9-bin.tar.gz

tar xzf apache-maven-3.3.9-bin.tar.gz
mkdir /usr/local/maven
mv apache-maven-3.3.9/ /usr/local/maven/
update-alternatives --install /usr/bin/mvn mvn /usr/local/maven/apache-maven-3.3.9/bin/mvn 1
update-ca-certificates -f

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
python3 -m pip install --upgrade pip

python3 -m pip install --upgrade --ignore-installed nose cpplint==1.3.0 pylint==2.3.1 nose-timer 'numpy<2.0.0,>1.16.0' 'requests<3,>=2.20.0' scipy boto3

# CMake 3.13.2+ is required
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar -zxvf cmake-3.16.5.tar.gz
cd cmake-3.16.5
./bootstrap
make -j$(nproc)
sudo make install

