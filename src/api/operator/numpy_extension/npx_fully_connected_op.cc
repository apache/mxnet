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
 * \file npx_fully_connected_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_fully_connected_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/fully_connected-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.fully_connected")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  int args_size = args.size();
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_fully_connected");
  op::FullyConnectedParam param;
  // no_bias
  param.no_bias = args[args_size - 2].operator bool();
  // inputs
  int num_inputs = 2;
  if (param.no_bias) {
    num_inputs = 2;
  } else {
    num_inputs = 3;
  }
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // num_hidden
  param.num_hidden = args[args_size - 3].operator int();
  // flatten
  param.flatten = args[args_size - 1].operator bool();

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::FullyConnectedParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
