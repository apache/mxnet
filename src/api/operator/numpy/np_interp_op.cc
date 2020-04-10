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
 * \file np_interp_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_interp_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_interp_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.interp")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_interp");
  nnvm::NodeAttrs attrs;
  op::NumpyInterpParam param;
  if (args[3].type_code() == kNull) {
    param.left = dmlc::nullopt;
  } else {
    param.left = args[3].operator double();
  }
  if (args[4].type_code() == kNull) {
    param.right = dmlc::nullopt;
  } else {
    param.right = args[4].operator double();
  }
  if (args[5].type_code() == kNull) {
    param.period = dmlc::nullopt;
  } else {
    param.period = args[5].operator double();
  }
  if (args[2].type_code() == kDLInt || args[2].type_code() == kDLFloat) {
    param.x_scalar = args[2].operator double();
    param.x_is_scalar = true;
    attrs.op = op;
    attrs.parsed = std::move(param);
    SetAttrDict<op::NumpyInterpParam>(&attrs);
    NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
    int num_inputs = 2;
    int num_outputs = 0;
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  } else {
    param.x_scalar = 0.0;
    param.x_is_scalar = false;
    attrs.op = op;
    attrs.parsed = std::move(param);
    SetAttrDict<op::NumpyInterpParam>(&attrs);
    NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*(),
                         args[2].operator mxnet::NDArray*()};
    int num_inputs = 3;
    int num_outputs = 0;
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

}  // namespace mxnet
