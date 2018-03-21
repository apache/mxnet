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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

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