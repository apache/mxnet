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
 * \file np_cumsum.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_triu_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_triu_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.triu")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  op::TriuParam param;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npi_triu");
  // inputs
  param.k = args[1].operator int();
  NDArray* inputs[] = {args[0].operator NDArray*()};

  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::TriuParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, 1, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
