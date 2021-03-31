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
 * \file npx_one_hot_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy_extension/npx_one_hot_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/indexing_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npx.one_hot")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npx_one_hot");
  op::OneHotParam param;
  // inputs
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  // depth
  param.depth = args[1].operator int64_t();
  // on_value
  if (args[2].type_code() == kNull) {
    param.on_value = 1.0;
  } else {
    param.on_value = args[2].operator double();
  }
  // off_value
  if (args[3].type_code() == kNull) {
    param.off_value = 0.0;
  } else {
    param.off_value = args[3].operator double();
  }
  // dtype
  if (args[4].type_code() != kNull) {
    param.dtype = String2MXNetTypeWithBool(args[4].operator std::string());
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::OneHotParam>(&attrs);
  int num_outputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
