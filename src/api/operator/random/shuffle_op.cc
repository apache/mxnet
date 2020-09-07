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
 * \file shuffle_op.cc
 * \brief Implementation of the API of functions in src/operator/random/shuffle_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/elemwise_op_common.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.shuffle")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_shuffle");
  nnvm::NodeAttrs attrs;

  NDArray* inputs[1];
  int num_inputs = 1;

  if (args[0].type_code() != kNull) {
    inputs[0] = args[0].operator mxnet::NDArray *();
  }

  attrs.op = op;

  NDArray* out = args[1].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(1);
  } else {
    *ret = ndoutputs[0];
  }
});

}  // namespace mxnet
