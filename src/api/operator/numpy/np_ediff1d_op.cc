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
 * \file np_ediff1d_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_ediff1d_op.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/numpy/np_ediff1d_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.ediff1d")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_ediff1d");
  nnvm::NodeAttrs attrs;
  op::EDiff1DParam param;
  int num_inputs = 1;
  NDArray* inputs[3];
  inputs[0] = args[0].operator mxnet::NDArray*();
  // the order of `to_end` and `to_begin` array in the backend is different from the front-end
  if (args[2].type_code() == kDLFloat || args[2].type_code() == kDLInt) {
    param.to_begin_scalar = args[2].operator double();
    param.to_begin_arr_given = false;
  } else if (args[2].type_code() == kNull) {
    param.to_begin_scalar = dmlc::nullopt;
    param.to_begin_arr_given = false;
  } else {
    param.to_begin_scalar = dmlc::nullopt;
    param.to_begin_arr_given = true;
    inputs[num_inputs] = args[2].operator mxnet::NDArray*();
    num_inputs++;
  }

  if (args[1].type_code() == kDLFloat || args[1].type_code() == kDLInt) {
    param.to_end_scalar = args[1].operator double();
    param.to_end_arr_given = false;
  } else if (args[1].type_code() == kNull) {
    param.to_end_scalar = dmlc::nullopt;
    param.to_end_arr_given = false;
  } else {
    param.to_end_scalar = dmlc::nullopt;
    param.to_end_arr_given = true;
    inputs[num_inputs] = args[1].operator mxnet::NDArray*();
    num_inputs++;
  }

  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::EDiff1DParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
