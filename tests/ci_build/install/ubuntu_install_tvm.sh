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

# Build and install TVM
cd /tmp
git clone https://github.com/dmlc/tvm/ --recursive
cd tvm
cp make/config.mk
echo USE_CUDA=1 >> config.mk
echo LLVM_CONFIG=llvm-config-5.0 >> config.mk
echo USE_RPC=1 >> config.mk
echo USE_GRAPH_RUNTIME=1 >> config.mk
echo CUDA_PATH=/usr/local/cuda >> config.mk
make -j10

cd python
python setup.py install
cd -

cd topi/python
python setup.py install
cd -
