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

CCACHE_TAG="v4.6.2"

export DEBIAN_FRONTEND=noninteractive

apt update || true
apt install -y \
    libxslt1-dev \
    docbook-xsl \
    xsltproc \
    libxml2-utils

apt install -y --no-install-recommends \
    autoconf \
    asciidoc \
    xsltproc

mkdir -p /work/deps/ccache
cd /work/deps/ccache

git clone --recursive -b $CCACHE_TAG https://github.com/ccache/ccache.git ccache
cd ccache
./autogen.sh
./configure
make -j$(nproc)
make install

popd
rm -rf /work/deps/ccache

