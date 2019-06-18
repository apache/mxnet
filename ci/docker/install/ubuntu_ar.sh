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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

wget https://mirror.clarkson.edu/gnu/binutils/binutils-2.27.tar.gz

export DEBIAN_FRONTEND=noninteractive
apt-get update || true
apt-get install -y \
    wget

mkdir /opt/binutils_install && mkdir /opt/binutils_install && mkdir /opt/binutils && cd /opt/binutils
wget -nv https://mirror.clarkson.edu/gnu/binutils/binutils-2.27.tar.gz
tar -xvf binutils-2.27.tar.gz && cd binutils-2.27
./configure --prefix=/opt/binutils_other --exec-prefix=/opt/binutils_install
make -j$(nproc)
make install
ln -s /opt/binutils_install/bin/ar /usr/local/bin/ar
