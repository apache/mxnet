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

CCACHE_TAG="v3.7.12"

yum install -y git autoconf gcc make gperf

pushd .
mkdir -p /work/deps
cd /work/deps
git clone --recursive -b $CCACHE_TAG https://github.com/ccache/ccache.git ccache
cd ccache
./autogen.sh
./configure --disable-man --prefix=/usr/local
make -j $(nproc)
make install
cd /work/deps
rm -rf /work/deps/ccache
popd

