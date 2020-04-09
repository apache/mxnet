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
 * \file matrix_op.cc
 * \brief Implementation of the API of functions in src/operator/tensor/matrix_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/matrix_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.clip")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_clip");
  nnvm::NodeAttrs attrs;
  op::ClipParam param;
  NDArray* inputs[1];

  if (args[0].type_code() != kNull) {
    inputs[0] = args[0].operator mxnet::NDArray *();
  }

  if (args[1].type_code() != kNull) {
    param.a_min = args[1].operator double();
  } else {
    param.a_min = -INFINITY;
  }

  if (args[2].type_code() != kNull) {
    param.a_max = args[2].operator double();
  } else {
    param.a_max = INFINITY;
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::ClipParam>(&attrs);

  NDArray* out = args[3].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  // set the number of outputs provided by the `out` arugment
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, 1, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(3);
  } else {
    *ret = ndoutputs[0];
  }
});

MXNET_REGISTER_API("_npi.tile")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tile");
  nnvm::NodeAttrs attrs;
  op::TileParam param;
  if (args[1].type_code() == kDLInt) {
    param.reps = Tuple<int>(1, args[1].operator int64_t());
  } else {
  param.reps = Tuple<int>(args[1].operator ObjectRef());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::TileParam>(&attrs);
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
