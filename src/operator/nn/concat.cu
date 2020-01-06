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
 * Copyright (c) 2015 by Contributors
 * \file concat.cu
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"

namespace mxnet {
namespace op {

static void ConcatComputeExGPU(const nnvm::NodeAttrs& attrs,
                               const OpContext& op_ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  if (common::ContainsOnlyStorage(inputs, kCSRStorage) &&
      outputs[0].storage_type() == kCSRStorage) {
    ConcatCSRImpl<gpu>(attrs, op_ctx, inputs, req, outputs);
  } else {
    LogUnimplementedOp(attrs, op_ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(Concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ConcatComputeExGPU);

NNVM_REGISTER_OP(_rnn_param_concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ConcatComputeExGPU);

NNVM_REGISTER_OP(_backward_Concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

