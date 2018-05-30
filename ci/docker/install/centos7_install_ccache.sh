#!/bin/bash

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

yum -y install epel-release
yum -y install git
yum -y install autoconf
yum -y install wget
yum -y install make
yum -y install google-perftools
yum -y install asciidoc
yum -y install gcc-c++-4.8.*

cd /work

git clone --recursive -b v3.4.2 https://github.com/ccache/ccache.git

cd ccache

./autogen.sh
./configure
make -j$(nproc)
make install

popd

rm -rf /work/ccache

export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:=10G}
export CCACHE_DIR=${CCACHE_DIR:=/work/ccache}
