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
pushd .
cd ..
apt-get update
apt-get install -y automake libtool zip
git clone --recursive -b 3.5.1.1 https://github.com/google/protobuf.git

cd protobuf

# Default AMD64 protobuf target.
AMD64_PROTOBUF_TARGET=/usr/local

# Custom ARM protobuf target.
ARM_PROTOBUF_TARGET=/usr/aarch64-linux-gnu

# Install protoc 3.5 and build protobuf here (for onnx and onnx-tensorrt)
./autogen.sh
./configure --disable-shared CXXFLAGS=-fPIC --host=amd64 CC=gcc CXX=g++
make -j$(nproc)
make install

# Remove dynamic AMD64 protobuf libs to force linker to statically link
rm -rf $AMD64_PROTOBUF_TARGET/lib/libprotobuf-lite.so*
rm -rf $AMD64_PROTOBUF_TARGET/lib/libprotobuf.so*
rm -rf $AMD64_PROTOBUF_TARGET/lib/libprotoc.so*

mkdir -p /usr/local/protobuf/targets/aarch64-linux
make clean
./autogen.sh
./configure --disable-shared CXXFLAGS=-fPIC --host=arm-linux --with-protoc=/usr/local/bin/protoc --prefix=$ARM_PROTOBUF_TARGET
make -j$(nproc)
make install

# Remove dynamic ARM protobuf libs to force linker to statically link
rm -rf $ARM_PROTOBUF_TARGET/lib/libprotobuf-lite.so*
rm -rf $ARM_PROTOBUF_TARGET/lib/libprotobuf.so*
rm -rf $ARM_PROTOBUF_TARGET/lib/libprotoc.so*

ldconfig
popd
