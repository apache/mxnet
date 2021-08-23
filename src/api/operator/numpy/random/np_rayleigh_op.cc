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
 * \file np_rayleigh_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/random/np_rayleigh_op.h
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/random/np_rayleigh_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.rayleigh")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_rayleigh");
  op::NumpyRayleighParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  if (args[1].type_code() == kDLInt) {
      param.size = Tuple<index_t>(1, args[1].operator int64_t());
  } else if (args[1].type_code() == kNull) {
      param.size = dmlc::nullopt;
  } else {
      param.size = Tuple<index_t>(args[1].operator ObjectRef());
  }
  if (args[2].type_code() != kNull) {
    attrs.dict["ctx"] = args[2].operator std::string();
  }
  NDArray* out = args[3].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  NDArray* inputs[1];
  int num_inputs = 0;
  if (args[0].type_code() == kDLFloat || args[0].type_code() == kDLInt) {
    param.scale = args[0].operator double();
    num_inputs = 0;
  } else {
    param.scale = dmlc::nullopt;
    inputs[0] = args[0].operator mxnet::NDArray*();
    num_inputs = 1;
  }
  attrs.parsed = param;
  SetAttrDict<op::NumpyRayleighParam>(&attrs);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs,
                          &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(3);
  } else {
    *ret = ndoutputs[0];
  }
});

}  // namespace mxnet
