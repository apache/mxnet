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

# This script builds the static library of libpng that can be used as dependency of mxnet/opencv.
set -ex
PNG_VERSION=1.6.34
if [[ ! -f $DEPS_PATH/lib/libpng.a ]]; then
    # download and build libpng
    >&2 echo "Building libpng..."
    curl -s -L https://github.com/glennrp/libpng/archive/v$PNG_VERSION.zip -o $DEPS_PATH/libpng.zip
    unzip -q $DEPS_PATH/libpng.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/libpng-$PNG_VERSION/build
    pushd .
    cd $DEPS_PATH/libpng-$PNG_VERSION/build
    cmake \
          -D PNG_SHARED=OFF \
          -D PNG_STATIC=ON \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH \
          -D CMAKE_C_FLAGS=-fPIC ..
    $MAKE
    $MAKE install
    mkdir -p $DEPS_PATH/include/libpng
    ln -s $DEPS_PATH/include/png.h $DEPS_PATH/include/libpng/png.h
    popd
fi
