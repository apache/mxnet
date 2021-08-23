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
 * \file np_diff_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_diff.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/numpy/np_diff-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.diff")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_diff");
  nnvm::NodeAttrs attrs;
  op::DiffParam param;
  param.n = args[1].operator int();
  param.axis = args[2].operator int();

  // we directly copy DiffParam, which is trivially-copyable
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::DiffParam>(&attrs);

  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
