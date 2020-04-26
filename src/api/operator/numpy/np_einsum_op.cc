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
 * \file np_einsum_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_einsum_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../utils.h"
#include "../../../operator/numpy/np_einsum_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.einsum")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_einsum");
  nnvm::NodeAttrs attrs;
  op::NumpyEinsumParam param;
  int args_size = args.size();
  // param.num_args
  param.num_args = args_size - 3;
  // param.subscripts
  param.subscripts = args[args_size - 3].operator std::string();
  // param.optimize
  param.optimize = args[args_size - 1].operator int();

  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyEinsumParam>(&attrs);

  // inputs
  int num_inputs = param.num_args;
  std::vector<NDArray*> inputs_vec(num_inputs, nullptr);
  for (int i = 0; i < num_inputs; ++i) {
    inputs_vec[i] = args[i].operator mxnet::NDArray*();
  }
  NDArray** inputs = inputs_vec.data();

  // outputs
  NDArray* out = args[args_size - 2].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;

  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(args_size - 2);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

}  // namespace mxnet
