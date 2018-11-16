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

# This script builds the static library of lz4 that can be used as dependency of mxnet.
LZ4_VERSION=r130
if [[ ! -f $DEPS_PATH/lib/liblz4.a ]]; then
    # Download and build lz4
    >&2 echo "Building lz4..."
    curl -s -L https://github.com/lz4/lz4/archive/$LZ4_VERSION.zip -o $DEPS_PATH/lz4.zip
    unzip -q $DEPS_PATH/lz4.zip -d $DEPS_PATH
    cd $DEPS_PATH/lz4-$LZ4_VERSION
    make
    make PREFIX=$DEPS_PATH install
    cd -
fi
