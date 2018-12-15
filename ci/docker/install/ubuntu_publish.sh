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
apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test -y
add-apt-repository ppa:openjdk-r/ppa -y # Java lib
apt-get update
apt-get install -y git \
    cmake3 \
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
    pandoc \
    python3-pip \
    automake \
    pkg-config \
    openjdk-8-jdk
curl -o apache-maven-3.3.9-bin.tar.gz http://www.eu.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz  
tar xzf apache-maven-3.3.9-bin.tar.gz   
mkdir /usr/local/maven  
mv apache-maven-3.3.9/ /usr/local/maven/    
update-alternatives --install /usr/bin/mvn mvn /usr/local/maven/apache-maven-3.3.9/bin/mvn 1
update-ca-certificates -f
