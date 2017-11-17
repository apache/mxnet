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


# Exit script with error if any errors occur

echo "BUILD make"
cp make/config.mk .
echo "USE_CUDA=1" >> config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> config.mk
echo "USE_PROFILER=1" >> config.mk
echo "DEV=1" >> config.mk
echo "EXTRA_OPERATORS=example/ssd/operator" >> config.mk
echo "USE_CPP_PACKAGE=1" >> config.mk

set -e

make -j$(nproc) || exit -1

echo "BUILD cpp_test"
make -j$(nproc) test || exit -1
export MXNET_ENGINE_INFO=true
./build/tests/cpp/mxnet_test

export MXNET_ENGINE_INFO=false
export PYTHONPATH=$(pwd)/python

echo "BUILD python_test"
nosetests --verbose tests/python/unittest || exit -1
nosetests --verbose tests/python/gpu/test_operator_gpu.py || exit -1
nosetests --verbose tests/python/gpu/test_forward.py || exit -1
nosetests --verbose tests/python/train || exit -1

echo "BUILD python3_test"
nosetests3 --verbose tests/python/unittest || exit -1
nosetests3 --verbose tests/python/gpu/test_operator_gpu.py || exit -1
nosetests3 --verbose tests/python/gpu/test_forward.py || exit -1
nosetests3 --verbose tests/python/train || exit -1

echo "BUILD scala_test"
export PATH=$PATH:/opt/apache-maven/bin
make scalapkg || exit -1
make scalatest || exit -1

# echo "BUILD julia_test"
# export MXNET_HOME="${PWD}"
# /home/ubuntu/julia/bin/julia -e 'try Pkg.clone("MXNet"); catch end; Pkg.checkout("MXNet"); Pkg.build("MXNet"); Pkg.test("MXNet")' || exit -1
