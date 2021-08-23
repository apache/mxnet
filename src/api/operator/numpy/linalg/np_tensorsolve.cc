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
 * \file np_tensorsolve.cc
 * \brief Implementation of the API of functions in src/operator/numpy/linalg/np_tensorsolve.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/linalg/np_tensorsolve-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.tensorsolve")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tensorsolve");
  nnvm::NodeAttrs attrs;
  op::TensorsolveParam param;
  if (args[2].type_code() == kNull) {
    param.a_axes = Tuple<int>();
  } else {
    if (args[2].type_code() == kDLInt) {
      param.a_axes = Tuple<int>(1, args[2].operator int64_t());
    } else {
      param.a_axes = Tuple<int>(args[2].operator ObjectRef());
    }
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::TensorsolveParam>(&attrs);
  int num_inputs = 2;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
