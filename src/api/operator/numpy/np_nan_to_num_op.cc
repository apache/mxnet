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
 * \file np_nan_to_num_op.cc
 * \brief Implementation of the API of nan_to_num function in
 *        src/operator/tensor/np_elemwise_unary_op_basic.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/tensor/elemwise_unary_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.nan_to_num")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_nan_to_num");
  nnvm::NodeAttrs attrs;

  op::NumpyNanToNumParam param;
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};

  param.copy = args[1].operator bool();
  param.nan = args[2].operator double();

  if (args[3].type_code() == kNull) {
    param.posinf = dmlc::nullopt;
  } else {
    param.posinf = args[3].operator double();
  }

  if (args[4].type_code() == kNull) {
    param.neginf = dmlc::nullopt;
  } else {
    param.neginf = args[4].operator double();
  }

  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyNanToNumParam>(&attrs);

  NDArray* out = args[5].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  // set the number of outputs provided by the `out` arugment
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(5);
  } else {
    *ret = ndoutputs[0];
  }
});

}  // namespace mxnet
