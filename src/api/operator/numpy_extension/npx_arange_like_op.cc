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
 * \file npx_arange_like_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_arange_like_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/init_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.arange_like")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_arange_like");
  op::RangeLikeParam param;
  // inputs
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  // start
  if (args[1].type_code() == kNull) {
    param.start = 0.0;
  } else {
    param.start = args[1].operator double();
  }
  // step
  if (args[2].type_code() == kNull) {
    param.step = 1.0;
  } else {
    param.step = args[2].operator double();
  }
  // repeat
  if (args[3].type_code() == kNull) {
    param.repeat = 1;
  } else {
    param.repeat = args[3].operator int();
  }
  // ctx
  if (args[4].type_code() != kNull) {
    attrs.dict["ctx"] = args[4].operator std::string();
  }
  // axis
  if (args[5].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[5].operator int();
  }
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::RangeLikeParam>(&attrs);

  // outputs
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
