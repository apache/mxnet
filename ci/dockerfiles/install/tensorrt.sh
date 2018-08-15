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
apt-get install -y automake libtool
git clone --recursive -b 3.5.1.1 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure
make -j$(nproc)
make install
ldconfig
popd

# Install TensorRT
echo "TensorRT build enabled. Installing TensorRT."
wget -qO tensorrt.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0_1-1_amd64.deb
dpkg -i tensorrt.deb
apt-get update
apt-get install -y --allow-downgrades libnvinfer-dev
rm tensorrt.deb
