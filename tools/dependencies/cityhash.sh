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

# This script builds the static library of cityhash that can be used as dependency of mxnet.
set -ex
CITYHASH_VERSION=1.1.1
if [[ ! -f $DEPS_PATH/lib/libcityhash.a ]]; then
    # Download and build cityhash
    >&2 echo "Building cityhash..."
    git clone https://github.com/google/cityhash $DEPS_PATH/cityhash-$CITYHASH_VERSION
    pushd .
    cd $DEPS_PATH/cityhash-$CITYHASH_VERSION
    git reset --hard 8af9b8c2b889d80c22d6bc26ba0df1afb79a30db
    ./configure -prefix=$DEPS_PATH --enable-sse4.2
    $MAKE CXXFLAGS="-g -O3 -msse4.2"
    $MAKE install
    popd
fi
