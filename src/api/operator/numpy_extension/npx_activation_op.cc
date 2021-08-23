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
 * \file npx_activation_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_activation_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/activation-inl.h"

namespace mxnet {

inline int String2MXNetActType(const std::string& s) {
  using namespace op;
  if (s == "relu") {
    return activation::kReLU;
  } else if (s == "sigmoid") {
    return activation::kSigmoid;
  } else if (s == "log_sigmoid") {
    return activation::kLogSigmoid;
  } else if (s == "mish") {
    return activation::kMish;
  } else if (s == "tanh") {
    return activation::kTanh;
  } else if (s == "softrelu") {
    return activation::kSoftReLU;
  } else if (s == "softsign") {
    return activation::kSoftSign;
  } else {
    LOG(FATAL) << "unknown activation type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.activation")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_activation");
  op::ActivationParam param;
  // act_type
  param.act_type = String2MXNetActType(args[1].operator std::string());
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::ActivationParam>(&attrs);
  // inputs
  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
