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
 * \file npx_broadcast_like_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_broadcast_like_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/broadcast_reduce_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.broadcast_like")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_broadcast_like");
  op::BroadcastLikeParam param;
  // inputs
  int num_inputs = 2;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // lhs_axes
  if (args[2].type_code() == kNull) {
    param.lhs_axes = dmlc::optional<mxnet::TShape>();
  } else if (args[2].type_code() == kDLInt) {
    param.lhs_axes = TShape(1, args[2].operator int64_t());
  } else {
    param.lhs_axes = mxnet::TShape(args[2].operator ObjectRef());
  }
  // rhs_axes
  if (args[3].type_code() == kNull) {
    param.rhs_axes = dmlc::optional<mxnet::TShape>();
  } else if (args[3].type_code() == kDLInt) {
    param.rhs_axes = TShape(1, args[3].operator int64_t());
  } else {
    param.rhs_axes = mxnet::TShape(args[3].operator ObjectRef());
  }

  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::BroadcastLikeParam>(&attrs);

  // outputs
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
