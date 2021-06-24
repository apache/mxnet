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
 * \file npx_control_flow_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_control_flow_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/operator.h>
#include "../utils.h"
#include "../../../operator/npx_control_flow.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.foreach")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_foreach");
  op::NPXForeachParam param;
  int args_size = args.size();
  int num_inputs = args_size - 7;
  // inputs
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(args[0].value().v_handle);
  std::vector<std::shared_ptr<nnvm::Symbol> > subgraphs;
  subgraphs.push_back(std::make_shared<nnvm::Symbol>(*sym));
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 1; i < num_inputs + 1; ++i) {
    inputs.push_back(static_cast<mxnet::NDArray*>(args[i]));
  }

  param.num_args = num_inputs;
  param.num_outputs = args[1+num_inputs].operator int();
  param.num_out_data = args[2+num_inputs].operator int();
  if (args[3+num_inputs].type_code() == kDLInt) {
    param.in_state_locs = mxnet::Tuple<int64_t>(1, args[3+num_inputs].operator int64_t());
  } else {
    param.in_state_locs = mxnet::Tuple<int64_t>(args[3+num_inputs].operator ObjectRef());
  }
  if (args[4+num_inputs].type_code() == kDLInt) {
    param.in_data_locs = mxnet::Tuple<int64_t>(1, args[4+num_inputs].operator int64_t());
  } else {
    param.in_data_locs = mxnet::Tuple<int64_t>(args[4+num_inputs].operator ObjectRef());
  }
  if (args[5+num_inputs].type_code() == kDLInt) {
    param.remain_locs = mxnet::Tuple<int64_t>(1, args[5+num_inputs].operator int64_t());
  } else {
    param.remain_locs = mxnet::Tuple<int64_t>(args[5+num_inputs].operator ObjectRef());
  }
  if (args[6+num_inputs].type_code() == kDLInt) {
    param.in_state_index = mxnet::Tuple<int64_t>(1, args[6+num_inputs].operator int64_t());
  } else {
    param.in_state_index = mxnet::Tuple<int64_t>(args[6+num_inputs].operator ObjectRef());
  }
  attrs.parsed = param;
  attrs.op = op;
  attrs.subgraphs = subgraphs;
  SetAttrDict<op::NPXForeachParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  if (num_outputs == 1) {
    *ret = ndoutputs[0];
  } else {
    std::vector<NDArrayHandle> ndarray_handles;
    ndarray_handles.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      ndarray_handles.emplace_back(ndoutputs[i]);
    }
    *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
  }
});


MXNET_REGISTER_API("_npx.while_loop")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_while_loop");
  op::NPXWhileLoopParam param;
  int args_size = args.size();
  int num_inputs = args_size - 8;
  // inputs
  std::vector<std::shared_ptr<nnvm::Symbol> > subgraphs;
  subgraphs.reserve(2);
  for (int i = 0; i < 2; i++) {
    nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(args[i].value().v_handle);
    subgraphs.push_back(std::make_shared<nnvm::Symbol>(*sym));
  }
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 2; i < num_inputs + 2; ++i) {
    inputs.push_back(static_cast<mxnet::NDArray*>(args[i]));
  }

  param.num_args = num_inputs;
  param.max_iterations = args[2+num_inputs].operator int();
  if (args[3+num_inputs].type_code() == kDLInt) {
    param.cond_input_locs = mxnet::Tuple<int64_t>(1, args[3+num_inputs].operator int64_t());
  } else {
    param.cond_input_locs = mxnet::Tuple<int64_t>(args[3+num_inputs].operator ObjectRef());
  }
  if (args[4+num_inputs].type_code() == kDLInt) {
    param.func_input_locs = mxnet::Tuple<int64_t>(1, args[4+num_inputs].operator int64_t());
  } else {
    param.func_input_locs = mxnet::Tuple<int64_t>(args[4+num_inputs].operator ObjectRef());
  }
  if (args[5+num_inputs].type_code() == kDLInt) {
    param.func_var_locs = mxnet::Tuple<int64_t>(1, args[5+num_inputs].operator int64_t());
  } else {
    param.func_var_locs = mxnet::Tuple<int64_t>(args[5+num_inputs].operator ObjectRef());
  }
  param.num_out_data = args[6+num_inputs].operator int();
  param.num_outputs = args[7+num_inputs].operator int();
  attrs.parsed = param;
  attrs.op = op;
  attrs.subgraphs = subgraphs;
  SetAttrDict<op::NPXWhileLoopParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  if (num_outputs == 1) {
    *ret = ndoutputs[0];
  } else {
    std::vector<NDArrayHandle> ndarray_handles;
    ndarray_handles.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      ndarray_handles.emplace_back(ndoutputs[i]);
    }
    *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
  }
});

MXNET_REGISTER_API("_npx.cond")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_cond");
  op::NPXCondParam param;
  int args_size = args.size();
  int num_inputs = args_size - 7;
  // inputs
  std::vector<std::shared_ptr<nnvm::Symbol> > subgraphs;
  subgraphs.reserve(3);
  for (int i = 0; i < 3; i++) {
    nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(args[i].value().v_handle);
    subgraphs.push_back(std::make_shared<nnvm::Symbol>(*sym));
  }
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 3; i < num_inputs + 3; ++i) {
    inputs.push_back(static_cast<mxnet::NDArray*>(args[i]));
  }

  param.num_args = num_inputs;
  if (args[3+num_inputs].type_code() == kDLInt) {
    param.cond_input_locs = mxnet::Tuple<int64_t>(1, args[3+num_inputs].operator int64_t());
  } else {
    param.cond_input_locs = mxnet::Tuple<int64_t>(args[3+num_inputs].operator ObjectRef());
  }
  if (args[4+num_inputs].type_code() == kDLInt) {
    param.then_input_locs = mxnet::Tuple<int64_t>(1, args[4+num_inputs].operator int64_t());
  } else {
    param.then_input_locs = mxnet::Tuple<int64_t>(args[4+num_inputs].operator ObjectRef());
  }
  if (args[5+num_inputs].type_code() == kDLInt) {
    param.else_input_locs = mxnet::Tuple<int64_t>(1, args[5+num_inputs].operator int64_t());
  } else {
    param.else_input_locs = mxnet::Tuple<int64_t>(args[5+num_inputs].operator ObjectRef());
  }
  param.num_outputs = args[6+num_inputs].operator int();
  attrs.parsed = param;
  attrs.op = op;
  attrs.subgraphs = subgraphs;
  SetAttrDict<op::NPXCondParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  if (num_outputs == 1) {
    *ret = ndoutputs[0];
  } else {
    std::vector<NDArrayHandle> ndarray_handles;
    ndarray_handles.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      ndarray_handles.emplace_back(ndoutputs[i]);
    }
    *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
  }
});

}  // namespace mxnet
