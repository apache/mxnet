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

# Install Protobuf
# Install protoc 3.5 and build protobuf here (for onnx and onnx-tensorrt)
pushd .
cd ..
apt update
apt install -y automake libtool
git clone --recursive -b 3.5.1.1 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
CC=/usr/bin/gcc CXX=/usr/bin/g++ ./configure CXXFLAGS=-fPIC --disable-shared
make -j$(nproc)
make install
rm -rf /usr/local/lib/libprotobuf-lite.so*
rm -rf /usr/local/lib/libprotobuf.so*
rm -rf /usr/local/lib/libprotoc.so*
ldconfig
git clean -xdff
# install runtime for cross compilation
./autogen.sh
./configure --disable-shared --prefix=${CROSS_ROOT} --with-protoc=protoc CXXFLAGS=-fPIC --host=aarch64-unknown-linux-gnueabi
make -j$(nproc)
PREFIX=${CROSS_ROOT} make install
ldconfig
popd

