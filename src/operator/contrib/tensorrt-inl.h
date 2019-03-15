#ifndef MXNET_OPERATOR_CONTRIB_TENSORRT_INL_H_
#define MXNET_OPERATOR_CONTRIB_TENSORRT_INL_H_
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

/*!
 * Copyright (c) 2018 by Contributors
 * \file tensorrt-inl.h
 * \brief TensorRT Operator
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/serializer.h>
#include <dmlc/parameter.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>

#include <NvInfer.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <string>

#include "nnvm_to_onnx-inl.h"
#include "../operator_common.h"
#include "../../common/utils.h"
#include "../../common/serialization.h"
#include "../../executor/exec_pass.h"
#include "../../executor/graph_executor.h"
#include "../../executor/onnx_to_tensorrt.h"

namespace mxnet {
namespace op {

using namespace nnvm;
using int64 = ::google::protobuf::int64;


using trt_name_to_idx = std::map<std::string, uint32_t>;


struct TRTEngineParam {
  nvinfer1::IExecutionContext* trt_executor;
  std::vector<std::pair<uint32_t, nnvm_to_onnx::TypeIO> > binding_map;
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT

#endif  // MXNET_OPERATOR_CONTRIB_TENSORRT_INL_H_
