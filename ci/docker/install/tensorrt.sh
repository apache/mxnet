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

# Install protoc 3.5 and build protobuf here (for onnx and onnx-tensorrt)
# echo "TensorRT build enabled. Installing Protobuf."; \
# git clone --recursive -b 3.5.1.1 https://github.com/google/protobuf.git
# cd protobuf
# ./autogen.sh
# ./configure
# make -j$(nproc)
# make install
# ldconfig

# Install Protobuf
# Install protoc 3.5 and build protobuf here (for onnx and onnx-tensorrt)
pushd .
cd ..
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
wget -qO tensorrt.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-3.0.4-ga-cuda9.0_1.0-1_amd64.deb
dpkg -i tensorrt.deb
apt-get update
apt-get install -y --allow-downgrades libnvinfer-dev
rm tensorrt.deb

# Install ONNX
#pushd .
#cd 3rdparty/onnx-tensorrt/third_party/onnx
#rm -rf build
#mkdir build
#cd build
#cmake -DCMAKE_CXX_FLAGS=-I/usr/include/python${PYVER} -DBUILD_SHARED_LIBS=ON ..
#make -j$(nproc)
#make install
#ldconfig
#cd ..
#mkdir /usr/include/x86_64-linux-gnu/onnx
#cp build/onnx/onnx*pb.* /usr/include/x86_64-linux-gnu/onnx
#cp build/libonnx.so /usr/local/lib
#ldconfig
#popd
#
## Install ONNX-TensorRT
#echo "==============================================================="
#pwd
#ls -la
#cd ..
#ls -la
#cd /
#ls -R 
#echo "==============================================================="
#cd 3rdparty/onnx-tensorrt/
#mkdir build
#cd build
#cmake ..
#make -j$(nproc)
#make install
#ldconfig
