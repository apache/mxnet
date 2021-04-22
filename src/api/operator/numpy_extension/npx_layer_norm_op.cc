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
 * \file npx_layer_norm_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_layer_norm_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/layer_norm-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.layer_norm")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_layer_norm");
  op::LayerNormParam param;
  // inputs
  int num_inputs = 3;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // axis
  if (args[3].type_code() == kNull) {
    param.axis = -1;
  } else {
    param.axis = args[3].operator int();
  }
  // eps
  if (args[4].type_code() == kNull) {
    param.eps = 1e-5f;
  } else {
    param.eps = args[4].operator double();
  }
  // output_mean_var
  if (args[5].type_code() == kNull) {
    param.output_mean_var = false;
  } else {
    param.output_mean_var = args[5].operator bool();
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::LayerNormParam>(&attrs);
  int num_outputs = 3;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
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
