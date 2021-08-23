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

# This script builds the static library of libz that can be used as dependency of mxnet.
set -ex
ZLIB_VERSION=1.2.6
if [[ ! -f $DEPS_PATH/lib/libz.a ]]; then
    # Download and build zlib
    >&2 echo "Building zlib..."
    download \
        https://github.com/LuaDist/zlib/archive/${ZLIB_VERSION}.zip \
        ${DEPS_PATH}/zlib.zip
    unzip -q $DEPS_PATH/zlib.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/zlib-$ZLIB_VERSION/build
    pushd .
    cd $DEPS_PATH/zlib-$ZLIB_VERSION/build
    cmake \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH \
          -D BUILD_SHARED_LIBS=OFF ..
    $MAKE
    $MAKE install
    popd
fi
