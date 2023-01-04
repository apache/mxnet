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

if [ $# -lt 1 ]; then
    >&2 echo "Usage: build.sh <VARIANT> <BLAS>"
fi

export CURDIR=$PWD
export DEPS_PATH=$PWD/staticdeps
export VARIANT=$(echo $1 | tr '[:upper:]' '[:lower:]')
export PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
export BLAS=$(echo $2 | tr '[:upper:]' '[:lower:]')

if [[ $VARIANT == darwin* ]]; then
    export VARIANT="darwin"
fi

if [[ ! $BLAS ]]; then
    export BLAS="open"
fi

NUM_PROC=1
if [[ ! -z $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ ! -z $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
else
    >&2 echo "Can't discover number of cores."
fi
export NUM_PROC
>&2 echo "Using $NUM_PROC parallel jobs in building."

if [[ $DEBUG -eq 1 ]]; then
    export ADD_MAKE_FLAG="-j $NUM_PROC"
else
    export ADD_MAKE_FLAG="--quiet -j $NUM_PROC"
fi
export MAKE="make $ADD_MAKE_FLAG"

export CC="gcc"
export CXX="g++"
export CFLAGS="${CFLAGS:+${CFLAGS}} -fPIC -mno-avx"
export CXXFLAGS="${CXXFLAGS:+${CXXFLAGS}} -fPIC -mno-avx"
export FC="gfortran"
export PKG_CONFIG_PATH=$DEPS_PATH/lib/pkgconfig:$DEPS_PATH/lib64/pkgconfig:$DEPS_PATH/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$DEPS_PATH/include:$CPATH

mkdir -p $DEPS_PATH

# Build Dependencies
source tools/dependencies/make_shared_dependencies.sh

# Copy LICENSE
mkdir -p licenses
cp tools/dependencies/LICENSE.binary.dependencies licenses/
cp NOTICE licenses/
cp LICENSE licenses/

# Build mxnet
source tools/staticbuild/build_lib.sh
