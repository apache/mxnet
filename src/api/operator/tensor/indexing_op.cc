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
 * \file indexing_op.cc
 * \brief Implementation of the API of functions in src/operator/tensor/indexing_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/indexing_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.take")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_take");
  nnvm::NodeAttrs attrs;
  op::TakeParam param;
  NDArray* inputs[2];

  if (args[0].type_code() != kNull) {
    inputs[0] = args[0].operator mxnet::NDArray *();
  }

  if (args[1].type_code() != kNull) {
    inputs[1] = args[1].operator mxnet::NDArray *();
  }

  if (args[2].type_code() == kDLInt) {
    param.axis = args[2].operator int();
  }

  if (args[3].type_code() != kNull) {
    std::string mode = args[3].operator std::string();
    if (mode == "raise") {
      param.mode = op::take_::kRaise;
    } else if (mode == "clip") {
      param.mode = op::take_::kClip;
    } else if (mode == "wrap") {
      param.mode = op::take_::kWrap;
    }
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::TakeParam>(&attrs);

  NDArray* out = args[4].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  // set the number of outputs provided by the `out` arugment
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, 2, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(4);
  } else {
    *ret = ndoutputs[0];
  }
});

}  // namespace mxnet
