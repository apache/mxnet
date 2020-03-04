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
 * \file np_insert_op.cc
 * \brief Implementation of the API of functions in the following file:
          src/operator/numpy/np_insert_op_scalar.cc
          src/operator/numpy/np_insert_op_slice.cc
          src/operator/numpy/np_insert_op_tensor.cc
 */
#include "../utils.h"
#include "../../../operator/numpy/np_insert_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.insert_scalar")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_insert_scalar");
  nnvm::NodeAttrs attrs;
  op::NumpyInsertParam param;
  NDArray** inputs = NULL;
  int num_inputs;
  param.start = dmlc::nullopt;
  param.step = dmlc::nullopt;
  param.stop = dmlc::nullopt;
  if (args[1].type_code() == kDLInt ||
      args[1].type_code() == kDLUInt ||
      args[1].type_code() == kDLFloat) {
    param.val = args[1].operator double();
    inputs = new NDArray*[1]{args[0].operator mxnet::NDArray*()};
    num_inputs = 1;
  } else {
    param.val = dmlc::nullopt;
    inputs = new NDArray*[2]{args[0].operator mxnet::NDArray*(),
                             args[1].operator mxnet::NDArray*()};
    num_inputs = 2;
  }
  param.int_ind = args[2].operator int();
  if (args[3].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[3].operator int();
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  int num_outputs = 0;
  auto ndoutputs = Invoke<op::NumpyInsertParam>(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.insert_slice")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_insert_slice");
  nnvm::NodeAttrs attrs;
  op::NumpyInsertParam param;
  NDArray** inputs = NULL;
  int num_inputs;
  if (args[1].type_code() == kDLInt ||
      args[1].type_code() == kDLUInt ||
      args[1].type_code() == kDLFloat) {
    param.val = args[1].operator double();
    inputs = new NDArray*[1]{args[0].operator mxnet::NDArray*()};
    num_inputs = 1;
  } else {
    param.val = dmlc::nullopt;
    inputs = new NDArray*[2]{args[0].operator mxnet::NDArray*(),
                             args[1].operator mxnet::NDArray*()};
    num_inputs = 2;
  }
  if (args[2].type_code() == kNull) {
    param.start = dmlc::nullopt;
  } else {
    param.start = args[2].operator int();
  }
  if (args[3].type_code() == kNull) {
    param.stop = dmlc::nullopt;
  } else {
    param.stop = args[3].operator int();
  }
  if (args[4].type_code() == kNull) {
    param.step = dmlc::nullopt;
  } else {
    param.step = args[4].operator int();
  }
  if (args[5].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[5].operator int();
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  int num_outputs = 0;
  auto ndoutputs = Invoke<op::NumpyInsertParam>(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.insert_tensor")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_insert_tensor");
  nnvm::NodeAttrs attrs;
  op::NumpyInsertParam param;
  param.start = dmlc::nullopt;
  param.step = dmlc::nullopt;
  param.stop = dmlc::nullopt;
  NDArray** inputs = NULL;
  int num_inputs;
  if (args[2].type_code() == kDLInt ||
      args[2].type_code() == kDLUInt ||
      args[2].type_code() == kDLFloat) {
    param.val = args[2].operator double();
    inputs = new NDArray*[2]{args[0].operator mxnet::NDArray*(),
                             args[1].operator mxnet::NDArray*()};
    num_inputs = 2;
  } else {
    param.val = dmlc::nullopt;
    inputs = new NDArray*[3]{args[0].operator mxnet::NDArray*(),
                             args[1].operator mxnet::NDArray*(),
                             args[2].operator mxnet::NDArray*()};
    num_inputs = 3;
  }
  if (args[3].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[3].operator int();
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  int num_outputs = 0;
  auto ndoutputs = Invoke<op::NumpyInsertParam>(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
