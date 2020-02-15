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

set -eo pipefail

# This script builds the libraries of mxnet.
make_config=make/staticbuild/${PLATFORM}_${VARIANT}.mk
if [[ ! -f $make_config ]]; then
    >&2 echo "Couldn't find make config $make_config for the current settings."
    exit 1
fi

>&2 echo "Now building mxnet modules..."
cp $make_config config.mk

git submodule update --init --recursive || true

$MAKE DEPS_PATH=$DEPS_PATH DMLCCORE
$MAKE DEPS_PATH=$DEPS_PATH $PWD/3rdparty/tvm/nnvm/lib/libnnvm.a
$MAKE DEPS_PATH=$DEPS_PATH PSLITE
$MAKE DEPS_PATH=$DEPS_PATH mkldnn

>&2 echo "Now building mxnet..."
$MAKE DEPS_PATH=$DEPS_PATH

if [[ $PLATFORM == 'linux' ]]; then
    if [[ -f /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so ]]; then
        cp -L /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so lib/libgfortran.so.3
    elif [[ -f /usr/lib/x86_64-linux-gnu/libgfortran.so.3 ]]; then
        cp -L /usr/lib/x86_64-linux-gnu/libgfortran.so.3 lib/libgfortran.so.3
    else
        cp -L /usr/lib/x86_64-linux-gnu/libgfortran.so.4 lib/libgfortran.so.4
    fi
    cp -L /usr/lib/x86_64-linux-gnu/libquadmath.so.0 lib/libquadmath.so.0
fi

# Print the linked objects on libmxnet.so
>&2 echo "Checking linked objects on libmxnet.so..."
if [[ ! -z $(command -v readelf) ]]; then
    readelf -d lib/libmxnet.so
    strip --strip-unneeded lib/libmxnet.so
elif [[ ! -z $(command -v otool) ]]; then
    otool -L lib/libmxnet.so
    strip -u -r -x lib/libmxnet.so
else
    >&2 echo "Not available"
fi

if [[ ! -L deps ]]; then
    ln -s staticdeps deps
fi
