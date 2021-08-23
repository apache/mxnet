/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Copyright (c) 2015 by Contributors

#include <mxnet/io.h>
#include <dmlc/registry.h>
#include "./image_augmenter.h"
#include "./image_iter_common.h"

// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::DataIteratorReg);
DMLC_REGISTRY_ENABLE(::mxnet::DatasetReg);
DMLC_REGISTRY_ENABLE(::mxnet::BatchifyFunctionReg);
}  // namespace dmlc

namespace mxnet {
namespace io {
// Register parameters in header files
DMLC_REGISTER_PARAMETER(BatchParam);
DMLC_REGISTER_PARAMETER(BatchSamplerParam);
DMLC_REGISTER_PARAMETER(PrefetcherParam);
DMLC_REGISTER_PARAMETER(ImageNormalizeParam);
DMLC_REGISTER_PARAMETER(ImageRecParserParam);
DMLC_REGISTER_PARAMETER(ImageRecordParam);
DMLC_REGISTER_PARAMETER(ImageDetNormalizeParam);
}  // namespace io
}  // namespace mxnet
