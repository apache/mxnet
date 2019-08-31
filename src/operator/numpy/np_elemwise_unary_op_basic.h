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
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_unary_op_basic.h
 * \brief Function definition of elementwise unary operators
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_BASIC_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_BASIC_H_

#if MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP
#include <vector>
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

#if MXNET_USE_TVM_OP
template<const char* func>
void TVMOpExp2Compute(const nnvm::NodeAttrs& attrs,
                      const mxnet::OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].Size() == 0U) return;
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], outputs[0]});
}

template<const char* func>
void TVMExp2Backward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (inputs[0].Size() == 0U) return;
  using namespace mshadow;
  const TBlob& out_grad = inputs[0];
  const TBlob& out_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {out_grad, out_data, in_grad});
}
#endif  // MXNET_USE_TVM_OP

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_BASIC_H_
