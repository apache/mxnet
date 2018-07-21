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
pushd .
wget -nv https://api.github.com/repos/xianyi/OpenBLAS/git/refs/heads/master -O openblas_version.json
echo "Using openblas:"
cat openblas_version.json
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=ARMV8
make install
ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/libopenblas.a
ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/liblapack.a
popd
