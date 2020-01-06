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

#The script has been copied as is from a dockerfile that existed on previous MXNet versions (0.11)
#Written By: Ly

set -ex

apt-get update || true
apt-get -y install nodejs

git clone -b 1.38.6 https://github.com/kripken/emscripten.git
git clone -b 1.38.6 https://github.com/kripken/emscripten-fastcomp
cd emscripten-fastcomp
git clone -b 1.38.6 https://github.com/kripken/emscripten-fastcomp-clang tools/clang
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;JSBackend" \
-DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_EXAMPLES=OFF \
-DCLANG_INCLUDE_TESTS=OFF && make -j$(nproc)

chmod -R 777 /work/deps/emscripten-fastcomp/
