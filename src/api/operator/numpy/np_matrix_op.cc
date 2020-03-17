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

MXNET_REGISTER_API("_npi.split")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_split");
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  nnvm::NodeAttrs attrs;
  op::SplitParam param;
  param.axis = args[2].operator int();
  param.squeeze_axis = false;
  if (args[1].type_code() == kDLInt) {
    param.indices = TShape(0, 0);
    param.sections = args[1].operator int();
    CHECK_GT(param.sections, 0)
      << "ValueError: number sections must be larger than 0";
    CHECK_EQ(inputs[0]->shape()[param.axis] % param.sections, 0)
      << "ValueError: array split does not result in an equal division";
  } else {
    TShape t = TShape(args[1].operator ObjectRef());
    param.indices = TShape(t.ndim() + 1, 0);
    for (int i = 0; i < t.ndim(); ++i) {
      param.indices[i + 1] = t[i];
    }
    param.sections = 0;
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::SplitParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

}  // namespace mxnet
