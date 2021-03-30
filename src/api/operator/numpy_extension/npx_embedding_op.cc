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
 * \file npx_embedding_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_embedding_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/indexing_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.embedding")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_embedding");
  op::EmbeddingParam param;
  // inputs
  int num_inputs = 2;
  std::vector<NDArray*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  // input_dim
  param.input_dim = args[2].operator int64_t();
  // output_dim
  param.output_dim = args[3].operator int64_t();
  // dtype
  param.dtype = String2MXNetTypeWithBool(args[4].operator std::string());
  // sparse_grad;
  if (args[5].type_code() == kNull) {
    param.sparse_grad = false;
  } else {
    param.sparse_grad = args[5].operator bool();
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::EmbeddingParam>(&attrs);
  int num_outputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
