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
 * \file npx_pick_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_pick_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/broadcast_reduce_op.h"

namespace mxnet {

inline int String2PickMode(const std::string& s) {
  using namespace op;
  if (s == "wrap") {
    return kWrap;
  } else if (s == "clip") {
    return kClip;
  } else {
    LOG(FATAL) << "unknown mode type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.pick")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_pick");
  op::PickParam param;
  // axis
  if (args[2].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[2].operator int();
  }
  // mode
  param.mode = String2PickMode(args[3].operator std::string());
  // keepdims
  if (args[4].type_code() == kNull) {
    param.keepdims = false;
  } else {
    param.keepdims = args[4].operator bool();
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::PickParam>(&attrs);
  // inputs
  int num_inputs = 2;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < 2; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
