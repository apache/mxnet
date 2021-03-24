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
 * \file npx_batch_norm_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_batch_norm_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/batch_norm-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.batch_norm")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_batch_norm");
  op::BatchNormParam param;
  // eps
  param.eps = args[5].operator double();
  // momentum
  param.momentum = args[6].operator double();
  // fix_gamma
  param.fix_gamma = args[7].operator bool();
  // use_global_stats
  param.use_global_stats = args[8].operator bool();
  // output_mean_var
  param.output_mean_var = args[9].operator bool();
  // axis
  param.axis = args[10].operator int();
  // cudnn_off
  param.cudnn_off = args[11].operator bool();
  // min_calib_range
  if (args[12].type_code() == kDLFloat || args[12].type_code() == kDLInt) {
    param.min_calib_range = args[12].operator double();
  } else {
    param.min_calib_range = dmlc::nullopt;
  }
  // max_calib_range
  if (args[13].type_code() == kDLFloat || args[13].type_code() == kDLInt) {
    param.max_calib_range = args[13].operator double();
  } else {
    param.max_calib_range = dmlc::nullopt;
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::BatchNormParam>(&attrs);
  // inputs
  int num_inputs = 5;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  int num_outputs = 0;
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
