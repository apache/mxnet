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
 * \file deconvolution.cu
 * \brief
 * \author Wei Wu, Da Zheng
 */

#include "./deconvolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "../cudnn_ops.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "fully_connected-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

template <>
void DeconvolutionCompute<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  int dtype                       = inputs[0].type_flag_;
  CHECK_EQ(req.size(), 1);
  CHECK_EQ(req[deconv::kOut], kWriteTo);

#if MXNET_USE_CUDNN == 1
  STATIC_ASSERT_CUDNN_VERSION_GE(8000);
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    cudnn::ConvParam conv_param(param, false);
    bool ok =
        !param.cudnn_off &&
        cudnn::Exec<cudnn::ConvDgrad>(
            ctx, conv_param, inputs[deconv::kWeight], inputs[deconv::kData], outputs[deconv::kOut]);
    if (ok && !param.no_bias) {
      CHECK_EQ(inputs[deconv::kBias].shape_.ndim(), 1);
      auto layout = static_cast<mshadow::LayoutFlag>(param.layout.value());
      auto li     = cudnn::GetLayoutInfo(layout);
      if (li.channel_last ||
          !cudnn::LegacyAddBias(ctx, li, outputs[deconv::kOut], inputs[deconv::kBias])) {
        int k  = inputs[deconv::kBias].shape_.Size();
        auto b = inputs[deconv::kBias].reshape(cudnn::ExpandChannelDims(layout, k));
        BinaryBroadcastRTCCompute{"add"}(  // NOLINT(whitespace/braces)
            attrs,
            ctx,
            {outputs[deconv::kOut], b},
            {kWriteInplace},
            {outputs[deconv::kOut]});
      }
    }
    if (!ok) {
      if (!param.cudnn_off)
        LOG(WARNING)
            << "This deconvolution is not supported by cuDNN, MXNet deconvolution is applied.";
      DeconvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    DeconvolutionOp<gpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

template <>
void DeconvolutionGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob& out_grad             = inputs[0];
  const std::vector<TBlob>& in_grad = outputs;
  int dtype                         = out_grad.type_flag_;
  CHECK_EQ(req.size(), param.no_bias ? 2 : 3);
  CHECK_NE(req[deconv::kData], kWriteInplace);
  CHECK_NE(req[deconv::kWeight], kWriteInplace);
  if (!param.no_bias)
    CHECK_NE(req[deconv::kBias], kWriteInplace);

#if MXNET_USE_CUDNN == 1
  STATIC_ASSERT_CUDNN_VERSION_GE(8000);
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    cudnn::ConvParam conv_param(param, req[deconv::kData] == kAddTo);
    bool ok = !param.cudnn_off;
    ok      = ok &&
         (req[deconv::kData] == kNullOp ||
          cudnn::Exec<cudnn::Conv>(
              ctx, conv_param, inputs[0], inputs[1 + deconv::kWeight], outputs[deconv::kData]));
    conv_param.add_to = req[deconv::kWeight] == kAddTo;
    ok                = ok &&
         (req[deconv::kWeight] == kNullOp ||
          cudnn::Exec<cudnn::ConvWgrad>(
              ctx, conv_param, inputs[0], inputs[1 + deconv::kData], outputs[deconv::kWeight]));
    if (ok && !param.no_bias && req[deconv::kBias] != kNullOp) {
      auto li     = cudnn::GetLayoutInfo(static_cast<mshadow::LayoutFlag>(param.layout.value()));
      auto add_to = req[conv::kBias] == kAddTo;
      if (li.channel_last ||
          !cudnn::LegacyBiasGrad(ctx, li, add_to, outputs[deconv::kBias], inputs[0])) {
        if (li.channel_last) {
          // This kernel should be faster.
          auto y_grad = FlattenAs2DHead<gpu, DType>(inputs[0], ctx);
          AddBiasGrad(outputs[deconv::kBias], y_grad, req[deconv::kBias], param.num_filter, ctx);
        } else {
          TShape axes{static_cast<int>(li.ChannelIdx())};
          TShape small = ReduceAxesShapeImpl(
              inputs[0].shape_, dmlc::optional<mxnet::TShape>(axes), true, true);
          ReduceAxesRTCComputeImpl(ctx,
                                   {inputs[0]},
                                   {req[deconv::kBias]},
                                   {outputs[deconv::kBias]},
                                   small,
                                   "red::sum{}");
        }
      }
    }
    if (!ok) {
      if (!param.cudnn_off)
        LOG(WARNING)
            << "This deconvolution backward is not supported by cuDNN, MXNet op is applied.";
      DeconvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    DeconvolutionOp<gpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  })
#endif  // MXNET_USE_CUDNN
}

NNVM_REGISTER_OP(Deconvolution).set_attr<FCompute>("FCompute<gpu>", DeconvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_Deconvolution)
    .set_attr<FCompute>("FCompute<gpu>", DeconvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
