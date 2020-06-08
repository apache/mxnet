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
 * \file np_broadcast_reduce_op_boolean.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_broadcast_reduce_op_boolean.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_broadcast_reduce_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.all")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_all");
  nnvm::NodeAttrs attrs;
  op::NumpyReduceAxesBoolParam param;

  NDArray* out = args[3].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else if (args[1].type_code() == kDLInt) {
    param.axis = Tuple<int>(1, args[1].operator int64_t());
  } else {
    param.axis = Tuple<int>(args[1].operator ObjectRef());
  }
  param.keepdims = args[2].operator bool();
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyReduceAxesBoolParam>(&attrs);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(3);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

MXNET_REGISTER_API("_npi.any")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_any");
  nnvm::NodeAttrs attrs;
  op::NumpyReduceAxesBoolParam param;

  NDArray* out = args[3].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else if (args[1].type_code() == kDLInt) {
    param.axis = Tuple<int>(1, args[1].operator int64_t());
  } else {
    param.axis = Tuple<int>(args[1].operator ObjectRef());
  }
  param.keepdims = args[2].operator bool();
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyReduceAxesBoolParam>(&attrs);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(3);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

}  // namespace mxnet
