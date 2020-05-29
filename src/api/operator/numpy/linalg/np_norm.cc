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
 * \file np_norm.cc
 * \brief Implementation of the API of functions in src/operator/numpy/linalg/np_norm_forward.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/linalg/np_norm-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.norm")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npi_norm");
  op::NumpyNormParam param;
  param.ord = args[1].operator double();
  if (args[2].type_code() == kNull) {
    param.axis = dmlc::optional<mxnet::TShape>();
  } else {
    param.axis = mxnet::TShape(args[2].operator ObjectRef());
  }
  param.keepdims = args[3].operator bool();
  param.flag = args[4].operator int();

  attrs.op = op;
  attrs.parsed = std::move(param);
  SetAttrDict<op::NumpyNormParam>(&attrs);

  // inputs
  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;
  // outputs
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
