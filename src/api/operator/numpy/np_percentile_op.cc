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
 * \file np_percentile_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_percentile_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_percentile_op-inl.h"

namespace mxnet {

inline int String2MXNetPercentileType(const std::string& s) {
  using namespace op;
  if (s == "linear") {
    return percentile_enum::kLinear;
  } else if (s == "lower") {
    return percentile_enum::kLower;
  } else if (s == "higher") {
    return percentile_enum::kHigher;
  } else if (s == "midpoint") {
    return percentile_enum::kMidpoint;
  } else if (s== "nearest") {
    return percentile_enum::kNearest;
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npi.percentile")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_percentile");
  nnvm::NodeAttrs attrs;
  op::NumpyPercentileParam param;

  NDArray* out = args[5].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  if (args[2].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else if (args[2].type_code() == kDLInt) {
    param.axis = Tuple<int>(1, args[2].operator int64_t());
  } else {
    param.axis = Tuple<int>(args[2].operator ObjectRef());
  }
  param.interpolation = String2MXNetPercentileType(args[3].operator std::string());
  param.keepdims = args[4].operator bool();
  if (args[1].type_code() == kDLInt || args[1].type_code() == kDLFloat) {
    param.q_scalar = args[1].operator double();
    NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
    int num_inputs = 1;
    attrs.parsed = std::move(param);
    attrs.op = op;
    SetAttrDict<op::NumpyPercentileParam>(&attrs);
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
    if (out) {
      *ret = PythonArg(5);
    } else {
      *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
    }
  } else {
    param.q_scalar = dmlc::nullopt;
    NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
    int num_inputs = 2;
    attrs.parsed = std::move(param);
    attrs.op = op;
    SetAttrDict<op::NumpyPercentileParam>(&attrs);
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
    if (out) {
      *ret = PythonArg(5);
    } else {
      *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
    }
  }
});

}  // namespace mxnet
