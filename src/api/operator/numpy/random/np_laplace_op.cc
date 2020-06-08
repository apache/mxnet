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
 * \file np_laplace_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/random/np_laplace_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/random/np_laplace_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.laplace")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_laplace");
  nnvm::NodeAttrs attrs;
  op::NumpyLaplaceParam param;

  NDArray** inputs = new NDArray*[2]();
  int num_inputs = 0;

  if (args[0].type_code() == kNull) {
    param.loc = dmlc::nullopt;
  } else if (args[0].type_code() == kNDArrayHandle) {
    param.loc = dmlc::nullopt;
    inputs[num_inputs] = args[0].operator mxnet::NDArray *();
    num_inputs++;
  } else {
    param.loc = args[0].operator double();  // convert arg to T
  }

  if (args[1].type_code() == kNull) {
    param.scale = dmlc::nullopt;
  } else if (args[1].type_code() == kNDArrayHandle) {
    param.scale = dmlc::nullopt;
    inputs[num_inputs] = args[1].operator mxnet::NDArray *();
    num_inputs++;
  } else {
    param.scale = args[1].operator double();  // convert arg to T
  }

  if (args[2].type_code() == kNull) {
    param.size = dmlc::nullopt;
  } else {
    if (args[2].type_code() == kDLInt) {
      param.size = mxnet::Tuple<int>(1, args[2].operator int64_t());
    } else {
      param.size = mxnet::Tuple<int>(args[2].operator ObjectRef());
    }
  }

  if (args[3].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[3].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyLaplaceParam>(&attrs);
  if (args[4].type_code() != kNull) {
    attrs.dict["ctx"] = args[4].operator std::string();
  }

  inputs = inputs == nullptr ? nullptr : inputs;

  NDArray* out = args[5].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(5);
  } else {
    *ret = ndoutputs[0];
  }
});

}  // namespace mxnet
