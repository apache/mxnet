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

pushd .

cd /work/deps/

apt update
apt install -y automake libtool
git clone --recursive -b 1.4.0.rc0 https://github.com/apache/incubator-mxnet.git mxnet

cd mxnet/3rdparty/onnx-tensorrt/third_party/onnx
mkdir -p build
cd build
cmake \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CROSSCOMPILING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER=${CXX} \
    -DCMAKE_CXX_FLAGS=-I/usr/include/python${PYVER} \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=${CROSS_ROOT} \
    -G Ninja \
    ..

ninja -j 1 -v onnx/onnx.proto
ninja -j 1 -v
ninja install

cd /work/deps/mxnet/3rdparty/onnx-tensorrt/
mkdir -p build

cd build
cmake \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DCMAKE_CXX_FLAGS="-I ${cross_root}/include/ -I /usr/include/aarch64-linux-gnu" \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CROSSCOMPILING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${CROSS_ROOT} \
    -G Ninja \
    ..

ninja
ninja install

popd
