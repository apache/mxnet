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
 * \file np_fill_diagonal_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_fill_diagonal.cc */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/numpy/np_fill_diagonal_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.fill_diagonal")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_fill_diagonal");
  nnvm::NodeAttrs attrs;

  op::NumpyFillDiagonalParam param;
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};

  if (args[1].type_code() == kDLInt || args[1].type_code() == kDLUInt
      || args[1].type_code() == kDLFloat || args[1].type_code() == kDLBfloat) {
    param.val = Tuple<double>(1, args[1].operator double());
  } else {
    param.val = Obj2Tuple<double, Float>(args[1].operator ObjectRef());
  }
  param.wrap = args[2].operator bool();

  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyFillDiagonalParam>(&attrs);

  NDArray* out = args[3].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  // set the number of outputs provided by the `out` arugment
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(3);
  } else {
    *ret = ndoutputs[0];
  }
});

}  // namespace mxnet
