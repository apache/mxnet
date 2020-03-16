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

# Script to build ccache for debian and ubuntu based images.

set -ex

pushd .

dist=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [ $dist == "Ubuntu" ]; then
    zstd = "libzstd1-dev"
else  # Debian
    zstd = "libzstd-dev"
fi

apt update || true
apt install -y \
    $zstd \
    libb2-dev \
    autoconf \
    xsltproc

mkdir -p /work/deps
cd /work/deps

# Unset ARM toolchain cross-compilation configuration on dockcross
unset ARCH
unset DEFAULT_DOCKCROSS_IMAGE
unset CROSS_TRIPLE
unset CC
unset AS
unset AR
unset FC
unset CXX
unset CROSS_ROOT
unset CROSS_COMPILE
unset PKG_CONFIG_PATH
unset CMAKE_TOOLCHAIN_FILE
unset CPP
unset LD
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

git clone --recursive https://github.com/ccache/ccache.git
cd ccache
# Checkout a fixed & tested pre-release commit of ccache 4
# ccache 4 contains fixes for caching nvcc output: https://github.com/ccache/ccache/pull/381
git checkout 2e7154e67a5dd56852dae29d4c418d4ddc07c230

./autogen.sh
./configure --disable-man
make -j$(nproc)
make install

rm -rf /work/deps/ccache

popd
