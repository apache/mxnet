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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

# Multipackage installation does not fail in yum
yum -y install epel-release
yum -y install git
yum -y install wget
yum -y install atlas-devel # Provide clbas headerfiles
yum -y install openblas-devel
yum -y install lapack-devel
yum -y install opencv-devel
yum -y install protobuf-compiler
yum -y install protobuf-devel
yum -y install zeromq-devel
yum -y install openssl-devel
yum -y install gcc-c++-4.8.*
yum -y install make
yum -y install wget
yum -y install unzip
yum -y install ninja-build

# CMake 3.13.2+ is required
mkdir /opt/cmake && cd /opt/cmake
wget -nv https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.sh
sh cmake-3.13.5-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
rm cmake-3.13.5-Linux-x86_64.sh
cmake --version
