#! /bin/bash

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

DMLC_CORE=$(pwd)/../dmlc-core
cd ../nnvm/amalgamation
make clean
make DMLC_CORE_PATH=$DMLC_CORE nnvm.d
cp nnvm.d ../../amalgamation/
echo '#define MSHADOW_FORCE_STREAM

#ifndef MSHADOW_USE_CBLAS
#if (__MIN__ == 1)
#define MSHADOW_USE_CBLAS   0
#else
#define MSHADOW_USE_CBLAS   1
#endif
#endif
#define MSHADOW_USE_CUDA    0
#define MSHADOW_USE_MKL     0
#define MSHADOW_RABIT_PS    0
#define MSHADOW_DIST_PS     0
#define DMLC_LOG_STACK_TRACE 0

#include "mshadow/tensor.h"
#include "mxnet/base.h"
#include "dmlc/json.h"
#include "nnvm/tuple.h"
#include "mxnet/tensor_blob.h"' > temp
cat nnvm.cc >> temp
mv temp ../../amalgamation/nnvm.cc
