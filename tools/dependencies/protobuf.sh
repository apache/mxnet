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

# This script builds the static library of protobuf along with protoc, that can be used as dependency of mxnet.
set -ex
PROTOBUF_VERSION=3.5.1
if [[ $PLATFORM == 'darwin' ]]; then
    DY_EXT="dylib"
else
    DY_EXT="so"
fi

LIBPROTOBUF="$DEPS_PATH/lib/libprotobuf.$DY_EXT"
LIBPROTOC="$DEPS_PATH/lib/libprotoc.$DY_EXT"
if [[ ! -e $LIBPROTOBUF ]] || [[ ! -e $LIBPROTOC ]]; then
    # Download and build protobuf
    >&2 echo "Building protobuf..."
    curl -s -L https://github.com/google/protobuf/archive/v$PROTOBUF_VERSION.zip -o $DEPS_PATH/protobuf.zip
    unzip -q $DEPS_PATH/protobuf.zip -d $DEPS_PATH
    pushd .
    cd $DEPS_PATH/protobuf-$PROTOBUF_VERSION
    ./autogen.sh
    ./configure -prefix=$DEPS_PATH
    $MAKE
    $MAKE install
    popd
fi

