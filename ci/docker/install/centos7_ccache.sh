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

# Script to build ccache for centos7 based images

set -ex

pushd .

yum -y install autoconf libb2-devel libzstd-devel

mkdir -p /work/deps
cd /work/deps

git clone --recursive https://github.com/ccache/ccache.git
cd ccache
# Checkout a fixed & tested pre-release commit of ccache 4
# ccache 4 contains fixes for caching nvcc output: https://github.com/ccache/ccache/pull/381
git checkout 2e7154e67a5dd56852dae29d4c418d4ddc07c230

./autogen.sh
CXXFLAGS="-Wno-missing-field-initializers" ./configure --disable-man
make -j$(nproc)
make install

cd /work/deps
rm -rf /work/deps/ccache

popd
