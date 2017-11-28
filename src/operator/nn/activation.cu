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
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "../mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_activation-inl.h"
#endif

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1

template<typename DType>
static CuDNNActivationOp<DType> &get_cudnn_op(const ActivationParam& param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local CuDNNActivationOp<DType> cudnn_op;
#else
  static MX_THREAD_LOCAL CuDNNActivationOp<DType> cudnn_op;
#endif
  cudnn_op.Init(param);
  return cudnn_op;
}

template<>
void ActivationCompute<gpu>(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);

  // SoftReLU not supported by CUDNN yet
  if (param.act_type == activation::kSoftReLU) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      ActivationForward<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>(ctx,
          inputs[0], req[0], outputs[0]);
    });
  } else {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      get_cudnn_op<DType>(param).Forward(ctx, inputs[0], req[0], outputs[0]);
    });
  }
}

template<>
void ActivationGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);

  // SoftReLU not supported by CUDNN yet
  if (param.act_type == activation::kSoftReLU) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      ActivationBackward<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>(
          ctx, inputs[0], inputs[1], req[0], outputs[0]);
    });
  } else {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      get_cudnn_op<DType>(param).Backward(ctx, inputs[0], inputs[2], inputs[1], req[0], outputs[0]);
    });
  }
}
#endif

NNVM_REGISTER_OP(Activation)
.set_attr<FCompute>("FCompute<gpu>", ActivationCompute<gpu>);

NNVM_REGISTER_OP(_backward_Activation)
.set_attr<FCompute>("FCompute<gpu>", ActivationGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
