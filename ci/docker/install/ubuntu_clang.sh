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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex
# Install clang 3.9 (the same version as in XCode 8.*) and 5.0 (latest major release)
wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main" && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main" && \
    apt-get update && \
    apt-get install -y clang-3.9 clang-5.0 && \
    clang-3.9 --version && \
    clang-5.0 --version