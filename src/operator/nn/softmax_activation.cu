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
 * \file softmax_activation.cu
 * \brief
 * \author Junyuan Xie, Da Zheng
*/
#include "./softmax_activation-inl.h"
#include "../mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_softmax_activation-inl.h"
#endif

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<>
void SoftmaxActivationCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  const SoftmaxActivationParam& param = nnvm::get<SoftmaxActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  static thread_local CuDNNSoftmaxActivationOp op;
  op.Init(param);
  op.Forward(ctx, inputs[0], req[0], outputs[0]);
}

template<>
void SoftmaxActivationGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  const SoftmaxActivationParam& param = nnvm::get<SoftmaxActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);

  static thread_local CuDNNSoftmaxActivationOp op;
  op.Init(param);
  op.Backward(ctx, inputs[0], inputs[1], req[0], outputs[0]);
}
#endif

NNVM_REGISTER_OP(SoftmaxActivation)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxActivationCompute<gpu>);

NNVM_REGISTER_OP(_backward_SoftmaxActivation)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxActivationGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

