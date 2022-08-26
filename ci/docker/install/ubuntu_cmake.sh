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

apt update
apt install -y wget

mkdir -p /work/deps/cmake
cd /work/deps/cmake

CMAKE_VERSION=3.24.0
CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1,2)
CMAKE_ARCH=$(uname -m)
wget -q https://cmake.org/files/v$CMAKE_MAJOR/cmake-$CMAKE_VERSION-linux-$CMAKE_ARCH.sh
sh cmake-$CMAKE_VERSION-linux-$CMAKE_ARCH.sh --prefix=/usr/local --skip-license

cd /work/deps
rm -rf /work/deps/cmake

popd

