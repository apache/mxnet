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

# This script builds the static library of openblas that can be used as dependency of mxnet.
set +e # This script throws an error but otherwise works
set -x
OPENBLAS_VERSION=0.3.3
if [[ ! -e $DEPS_PATH/lib/libopenblas.a ]]; then
    # download and build openblas
    >&2 echo "Building openblas..."

    curl -s -L https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS_VERSION.zip -o $DEPS_PATH/openblas.zip
    unzip -q $DEPS_PATH/openblas.zip -d $DEPS_PATH
    pushd .
    cd $DEPS_PATH/OpenBLAS-$OPENBLAS_VERSION

    $MAKE DYNAMIC_ARCH=1 NO_SHARED=1 USE_OPENMP=1
    $MAKE PREFIX=$DEPS_PATH install
    popd
    ln -s libopenblas.a $DEPS_PATH/lib/libcblas.a
    ln -s libopenblas.a $DEPS_PATH/lib/liblapack.a
fi
