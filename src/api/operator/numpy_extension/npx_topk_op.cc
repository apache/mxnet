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
 * \file npx_topk_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_topk_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/ordering_op-inl.h"

namespace mxnet {

inline int String2ReturnType(const std::string& s) {
  using namespace op;
  if (s == "value") {
    return topk_enum::kReturnValue;
  } else if (s == "indices") {
    return topk_enum::kReturnIndices;
  } else if (s == "mask") {
    return topk_enum::kReturnMask;
  } else if (s == "both") {
    return topk_enum::kReturnBoth;
  } else {
    LOG(FATAL) << "unknown return type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.topk")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_topk");
  op::TopKParam param;
  // inputs
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray *()};
  // axis
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[1].operator int();
  }
  // k
  if (args[2].type_code() == kNull) {
    param.k = 1;
  } else {
    param.k = args[2].operator int();
  }
  // ret_typ
  param.ret_typ = String2ReturnType(args[3].operator std::string());
  // is_ascend
  if (args[4].type_code() == kNull) {
    param.is_ascend = false;
  } else {
    param.is_ascend = args[4].operator bool();
  }
  // dtype
  param.dtype = String2MXNetTypeWithBool(args[5].operator std::string());
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::TopKParam>(&attrs);
  int num_outputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  if (num_outputs == 1) {
    *ret = ndoutputs[0];
  } else {
    std::vector<NDArrayHandle> ndarray_handles;
    ndarray_handles.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      ndarray_handles.emplace_back(ndoutputs[i]);
    }
    *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
  }
});

}  // namespace mxnet
