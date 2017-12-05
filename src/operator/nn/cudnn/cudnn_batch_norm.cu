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
 * \file cudnn_batch_norm.cu
 * \brief
 * \author Junyuan Xie, Da Zheng
*/

#include "./cudnn_batch_norm-inl.h"
#include <vector>

namespace mxnet {
namespace op {
#if CUDNN_MAJOR == 4

template<typename DType>
static CuDNNBatchNormOp<DType> &GetCuDNNOp(const BatchNormParam& param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local CuDNNBatchNormOp<DType> op;
#else
  static MX_THREAD_LOCAL CuDNNBatchNormOp<DType> op;
#endif
  op.Init(param);
  return op;
}

static void BatchNormCompute_CuDNNv4(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
#if CUDNN_MAJOR >= 5
  LOG(FATAL) << "CuDNNBatchNorm is merged into BatchNorm for cudnn version above v5."
    "Use the later instead.";
#else
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 5U);
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> aux_states(inputs.begin() + 3, inputs.end());
  GetCuDNNOp<float>(param).Forward(ctx, in_data, req, outputs, aux_states);
#endif
}

static void BatchNormGradCompute_CuDNNv4(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
#if CUDNN_MAJOR >= 5
  LOG(FATAL) << "CuDNNBatchNorm is merged into BatchNorm for cudnn version above v5."
    "Use the later instead.";
#else
  CHECK_EQ(inputs.size(), 11U);
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  std::vector<TBlob> out_grad(1, inputs[0]);
  std::vector<TBlob> in_data(inputs.begin() + 3, inputs.begin() + 6);
  std::vector<TBlob> aux_states(inputs.begin() + 6, inputs.begin() + 8);
  std::vector<TBlob> out_data(inputs.begin() + 8, inputs.end());
  std::vector<TBlob> in_grad(outputs.begin(), outputs.begin() + 3);
  GetCuDNNOp<float>(param).Backward(ctx, out_grad, in_data, out_data,
      req, in_grad, aux_states);
#endif
}

NNVM_REGISTER_OP(CuDNNBatchNorm)
.set_attr<FCompute>("FCompute<gpu>", BatchNormCompute_CuDNNv4);

NNVM_REGISTER_OP(_backward_CuDNNBatchNorm)
.set_attr<FCompute>("FCompute<gpu>", BatchNormGradCompute_CuDNNv4);

#endif  // CUDNN_MAJOR == 4
}  // namespace op
}  // namespace mxnet

