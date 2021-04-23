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
 * \file npx_softmax_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_softmax_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/softmax-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.softmax")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  static const nnvm::Op* op = Op::Get("_npx_softmax");
  op::SoftmaxParam param;
  int args_size = args.size();
  // inputs
  int num_inputs = args_size - 4;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }

  // parse use_length
  if (args[args_size - 2].type_code() == kNull) {
    param.use_length = false;
  } else {
    param.use_length = args[args_size - 2].operator bool();
  }

  // parse axis
  if (args[args_size - 4].type_code() == kDLInt) {
    param.axis = args[args_size - 4].operator int();
  } else if (args[args_size - 4].type_code() == kDLFloat) {
    param.axis = static_cast<int>(args[args_size - 4].operator double());
  } else {
    param.axis = -1;
  }

  // parse temperature
  if (args[args_size - 3].type_code() == kNull) {
    param.temperature = dmlc::nullopt;
  } else {
    param.temperature = args[args_size - 3].operator double();
  }

  // parse dtype
  if (args[args_size - 1].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[args_size - 1].operator std::string());
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::SoftmaxParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npx.log_softmax")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  static const nnvm::Op* op = Op::Get("_npx_log_softmax");
  op::SoftmaxParam param;

  int args_size = args.size();
  // inputs
  int num_inputs = args_size - 4;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }

  // parse use_length
  if (args[args_size - 2].type_code() == kNull) {
    param.use_length = false;
  } else {
    param.use_length = args[args_size - 2].operator bool();
  }

  // parse axis
  if (args[args_size - 4].type_code() == kDLInt) {
    param.axis = args[args_size - 4].operator int();
  } else if (args[args_size - 4].type_code() == kDLFloat) {
    param.axis = static_cast<int>(args[args_size - 4].operator double());
  } else {
    param.axis = -1;
  }

  // parse temperature
  if (args[args_size - 3].type_code() == kNull) {
    param.temperature = dmlc::nullopt;
  } else {
    param.temperature = args[args_size - 3].operator double();
  }

  // parse dtype
  if (args[args_size - 1].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[args_size - 1].operator std::string());
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::SoftmaxParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npx.masked_softmax")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  static const nnvm::Op* op = Op::Get("_npx_masked_softmax");
  op::MaskedSoftmaxParam param;

  // inputs
  int num_inputs = 2;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // parse axis
  if (args[2].type_code() == kDLInt) {
    param.axis = args[2].operator int();
  } else if (args[2].type_code() == kDLFloat) {
    param.axis = static_cast<int>(args[2].operator double());
  } else {
    param.axis = -1;
  }
  // parse temperature
  if (args[3].type_code() == kNull) {
    param.temperature = dmlc::nullopt;
  } else {
    param.temperature = args[3].operator double();
  }
  // parse normalize
  if (args[4].type_code() == kNull) {
    param.normalize = true;
  } else {
    param.normalize = args[4].operator bool();
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::MaskedSoftmaxParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npx.masked_log_softmax")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  static const nnvm::Op* op = Op::Get("_npx_masked_log_softmax");
  op::MaskedSoftmaxParam param;

  // inputs
  int num_inputs = 2;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // parse axis
  if (args[2].type_code() == kDLInt) {
    param.axis = args[2].operator int();
  } else if (args[2].type_code() == kDLFloat) {
    param.axis = static_cast<int>(args[2].operator double());
  } else {
    param.axis = -1;
  }
  // parse temperature
  if (args[3].type_code() == kNull) {
    param.temperature = dmlc::nullopt;
  } else {
    param.temperature = args[3].operator double();
  }
  // parse normalize
  if (args[4].type_code() == kNull) {
    param.normalize = true;
  } else {
    param.normalize = args[4].operator bool();
  }

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::MaskedSoftmaxParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
