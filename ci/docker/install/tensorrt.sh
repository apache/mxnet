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

# Install gluoncv since we're testing Gluon models as well
pip2 install gluoncv==0.2.0
pip3 install gluoncv==0.2.0

# Install Protobuf
# Install protoc 3.5 and build protobuf here (for onnx and onnx-tensorrt)
pushd .
cd ..
apt-get update
apt-get install -y automake libtool zip
git clone --recursive -b 3.5.1.1 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure --disable-shared CXXFLAGS=-fPIC
make -j$(nproc)
make install
rm -rf /usr/local/lib/libprotobuf-lite.so*
rm -rf /usr/local/lib/libprotobuf.so*
rm -rf /usr/local/lib/libprotoc.so*
ldconfig
popd

# Install TensorRT
echo "TensorRT build enabled. Installing TensorRT."
apt-get update
apt-get install -y --allow-downgrades libnvinfer6=6.0.1-1+cuda10.1 \
        libnvinfer-dev=6.0.1-1+cuda10.1 \
        libcudnn7-dev=7.6.4.38-1+cuda10.1 \
        libcudnn7=7.6.4.38-1+cuda10.1 \
        libnvinfer-plugin-dev=6.0.1-1+cuda10.1 \
        libnvinfer-plugin6=6.0.1-1+cuda10.1
