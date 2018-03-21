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

######################################################################
# This script installs ONNX for Python along with all required dependencies 
# on a Ubuntu Machine.
# Tested on Ubuntu 16.04 distro.
######################################################################

set -e
set -x

echo "Installing libprotobuf-dev and protobuf-compiler ..."
apt-get install autoconf automake libtool curl make g++ unzip
git clone --recursive https://github.com/google/protobuf.git
cd protobuf
git checkout 80a37e0782d2d702d52234b62dd4b9ec74fd2c95
./autogen.sh
./configure --prefix=/usr
make
make check
make install 
ldconfig

echo "Installing pytest, pytest-cov, protobuf, Pillow, ONNX and tabulate ..."
pip2 install pytest==3.4.0 pytest-cov==2.5.1 protobuf==3.4.0 onnx==1.1.0 Pillow==5.0.0 tabulate==0.7.5
pip3 install pytest==3.4.0 pytest-cov==2.5.1 protobuf==3.4.0 onnx==1.1.0 Pillow==5.0.0 tabulate==0.7.5
