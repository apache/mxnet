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
 * \file np_matrix_op.cc
 * \brief Implementation of the API of functions in src/operator/tensor/matrix_op.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/tensor/matrix_op-inl.h"
#include "../../../operator/numpy/np_matrix_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.expand_dims")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_expand_dims");
  nnvm::NodeAttrs attrs;
  op::ExpandDimParam param;
  param.axis = args[1].operator int();

  // we directly copy ExpandDimParam, which is trivially-copyable
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::ExpandDimParam>(&attrs);

  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, 1, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.column_stack")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_column_stack");
  nnvm::NodeAttrs attrs;
  op::NumpyColumnStackParam param;
  param.num_args = args.size();

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::NumpyColumnStackParam>(&attrs);
  int num_outputs = 0;
  NDArray** inputs = new NDArray*[param.num_args];
  for (int i = 0; i < param.num_args; ++i) {
    inputs[i] = args[i].operator mxnet::NDArray*();
  }
  auto ndoutputs = Invoke(op, &attrs, param.num_args, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
  delete[] inputs;
});

}  // namespace mxnet
