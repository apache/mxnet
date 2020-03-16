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

apt update || true
apt install -y \
    autoconf \
    gperf \
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
git checkout v3.7.8
# Backport cuda related fixes: https://github.com/ccache/ccache/pull/381
git config user.name "MXNet CI"
git config user.email "MXNetCI@example.com"
git cherry-pick --strategy-option=theirs c4fffda031034f930df2cf188878b8f9160027df
git cherry-pick 0dec5c2df3e3ebc1fbbf33f74c992bef6264f37a

./autogen.sh
./configure --disable-man
make -j$(nproc)
make install

rm -rf /work/deps/ccache

popd
