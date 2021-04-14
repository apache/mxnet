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
 * \file npx_batch_dot_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_batch_dot_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/dot-inl.h"

namespace mxnet {

inline int String2ForwardStype(const std::string& s) {
  using namespace op;
  if (s == "default") {
    return kDefaultStorage;
  } else if (s == "row_sparse") {
    return kRowSparseStorage;
  } else if (s == "csr") {
    return kCSRStorage;
  } else {
    LOG(FATAL) << "unknown forward storage type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.batch_dot")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_batch_dot");
  op::DotParam param;
  // inputs
  int num_inputs = 2;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // transpose_a
  if (args[2].type_code() == kNull) {
    param.transpose_a = false;
  } else {
    param.transpose_a = args[2].operator bool();
  }
  // transpose_b
  if (args[3].type_code() == kNull) {
    param.transpose_b = false;
  } else {
    param.transpose_b = args[3].operator bool();
  }
  // forward_stype
  if (args[4].type_code() == kNull) {
    param.forward_stype = dmlc::nullopt;
  } else {
    param.forward_stype = String2ForwardStype(args[4].operator std::string());
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::DotParam>(&attrs);
  int num_outputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
