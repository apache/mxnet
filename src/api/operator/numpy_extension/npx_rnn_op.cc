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
 * \file npx_rnn_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_rnn_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/rnn-inl.h"

namespace mxnet {

inline int String2ComputeMode(const std::string& s) {
  using namespace op;
  if (s == "rnn_relu") {
    return rnn_enum::kRnnRelu;
  } else if (s == "rnn_tanh") {
    return rnn_enum::kRnnTanh;
  } else if (s == "lstm") {
    return rnn_enum::kLstm;
  } else if (s == "gru") {
    return rnn_enum::kGru;
  } else {
    LOG(FATAL) << "unknown compute mode " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.rnn")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_rnn");
  op::RNNParam param;
  int args_size = args.size();
  int num_inputs = 0;

  // mode
  param.mode = String2ComputeMode(args[args_size - 7].operator std::string());
  num_inputs = (param.mode == op::rnn_enum::kLstm) ? 4 : 3;
  // use_sequence_length
  if (args[args_size - 5].type_code() == kNull) {
    param.use_sequence_length = false;
  } else {
    param.use_sequence_length = args[args_size - 5].operator bool();
  }
  if (param.use_sequence_length) num_inputs += 1;
  // inputs
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // state_size
  param.state_size = (uint32_t) (args[args_size - 11].operator int());
  // num_layers
  param.num_layers = (uint32_t) (args[args_size - 10].operator int());
  // bidirectional
  if (args[args_size - 9].type_code() == kNull) {
    param.bidirectional = false;
  } else {
    param.bidirectional = args[args_size - 9].operator bool();
  }
  // state_outputs
  if (args[args_size - 8].type_code() == kNull) {
    param.state_outputs = false;
  } else {
    param.state_outputs = args[args_size - 8].operator bool();
  }
  // p
  if (args[args_size - 6].type_code() == kNull) {
    param.p = 0.0;
  } else {
    param.p = args[args_size - 6].operator double();
  }
  // projection_size
  if (args[args_size - 4].type_code() == kNull) {
    param.projection_size = dmlc::nullopt;
  } else {
    param.projection_size = args[args_size - 4].operator int();
  }
  // lstm_state_clip_min
  if (args[args_size - 3].type_code() == kNull) {
    param.lstm_state_clip_min = dmlc::nullopt;
  } else {
    param.lstm_state_clip_min = args[args_size - 3].operator double();
  }
  // lstm_state_clip_max
  if (args[args_size - 2].type_code() == kNull) {
    param.lstm_state_clip_max = dmlc::nullopt;
  } else {
    param.lstm_state_clip_max = args[args_size - 2].operator double();
  }
  // lstm_state_clip_nan
  if (args[args_size - 1].type_code() == kNull) {
    param.lstm_state_clip_nan = false;
  } else {
    param.lstm_state_clip_nan = args[args_size - 1].operator bool();
  }
  // initialize
  param.seq_length_ = 0;
  param.batch_size_ = 0;
  param.input_size_ = 0;
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::RNNParam>(&attrs);
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
