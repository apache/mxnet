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
 * \file npx_dropout_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_dropout_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/nn/dropout-inl.h"

namespace mxnet {

inline int String2Mode(const std::string& s) {
  using namespace op;
  if (s == "training") {
    return dropout::kTraining;
  } else if (s == "always") {
    return dropout::kAlways;
  } else {
    LOG(FATAL) << "unknown dropout mode " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.dropout")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_dropout");
  op::DropoutParam param;
  // inputs
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  // p
  param.p = args[1].operator double();
  // mode
  param.mode = String2Mode(args[2].operator std::string());
  // axes
  if (args[3].type_code() == kNull) {
    param.axes = TShape(0, 0);
  } else if (args[3].type_code() == kDLInt) {
    param.axes = TShape(1, args[3].operator int64_t());
  } else {
    param.axes = TShape(args[3].operator ObjectRef());
  }
  // cudnn_off
  if (args[4].type_code() == kNull) {
    param.cudnn_off = false;
  } else {
    param.cudnn_off = args[4].operator bool();
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::DropoutParam>(&attrs);
  int num_outputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
