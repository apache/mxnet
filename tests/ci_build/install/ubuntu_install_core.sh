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
set -e
set -x

# install libraries for building mxnet c++ core on ubuntu

apt-get update && apt-get install -y \
    build-essential git libopenblas-dev liblapack-dev libopencv-dev \
    libcurl4-openssl-dev cmake wget unzip sudo ninja-build

# Link Openblas to Cblas as this link does not exist on ubuntu16.04
ln -s /usr/lib/libopenblas.so /usr/lib/libcblas.so
