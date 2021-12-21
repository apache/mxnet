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
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
 */

#include "./convolution-inl.h"
#include <vector>
#include "./depthwise_convolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "../cudnn_ops.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "fully_connected-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

template <>
void ConvolutionCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  int dtype                     = inputs[conv::kData].type_flag_;
  CHECK_EQ(req.size(), 1);
  CHECK_EQ(req[conv::kOut], kWriteTo);

#if MXNET_USE_CUDNN == 1
  STATIC_ASSERT_CUDNN_VERSION_GE(8000);
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    cudnn::ConvParam conv_param(param, false);
    bool ok = !param.cudnn_off &&
              cudnn::Exec<cudnn::Conv>(
                  ctx, conv_param, inputs[conv::kData], inputs[conv::kWeight], outputs[conv::kOut]);
    if (ok && !param.no_bias) {
      CHECK_EQ(inputs[conv::kBias].shape_.ndim(), 1);
      auto layout = static_cast<mshadow::LayoutFlag>(param.layout.value());
      auto li     = cudnn::GetLayoutInfo(layout);
      if (li.channel_last ||
          !cudnn::LegacyAddBias(ctx, li, outputs[conv::kOut], inputs[conv::kBias])) {
        int k  = inputs[conv::kBias].shape_.Size();
        auto b = inputs[conv::kBias].reshape(cudnn::ExpandChannelDims(layout, k));
        BinaryBroadcastRTCCompute{"add"}(  // NOLINT(whitespace/braces)
            attrs,
            ctx,
            {outputs[conv::kOut], b},
            {kWriteInplace},
            {outputs[conv::kOut]});
      }
    }
    if (!ok) {
      if (!param.cudnn_off)
        LOG(WARNING) << "This convolution is not supported by cuDNN, MXNet convolution is applied.";
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  if (param.layout.value() != kNCW && param.layout.value() != kNCHW &&
      param.layout.value() != kNCDHW) {
    // Need CuDNN > 5.0 for layout support. use MXNet implementation
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    })
    return;
  }

  if (param.num_filter == param.num_group && param.layout.value() == mshadow::kNCHW &&
      param.num_filter == inputs[conv::kData].shape_[1] && param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) && dtype == mshadow::kFloat32) {
    mxnet::ShapeVector in_shape(inputs.size());
    mxnet::ShapeVector out_shape(1, outputs[0].shape_);
    for (size_t i = 0; i < in_shape.size(); i++)
      in_shape[i] = inputs[i].shape_;
    DepthwiseConvolutionOp<float> op;
    op.Init(param, in_shape, out_shape);
    op.Forward(ctx, inputs, req, outputs);
    return;
  }

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    ConvolutionOp<gpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

template <>
void ConvolutionGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob& out_grad             = inputs[0];
  const std::vector<TBlob>& in_grad = outputs;
  int dtype                         = out_grad.type_flag_;
  CHECK_EQ(req.size(), param.no_bias ? 2 : 3);
  CHECK_NE(req[conv::kData], kWriteInplace);
  CHECK_NE(req[conv::kWeight], kWriteInplace);
  if (!param.no_bias)
    CHECK_NE(req[conv::kBias], kWriteInplace);

#if MXNET_USE_CUDNN == 1
  STATIC_ASSERT_CUDNN_VERSION_GE(8000);
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    cudnn::ConvParam conv_param(param, req[conv::kData] == kAddTo);
    bool ok           = !param.cudnn_off;
    ok                = ok && (req[conv::kData] == kNullOp ||
                cudnn::Exec<cudnn::ConvDgrad>(
                    ctx, conv_param, inputs[1 + conv::kWeight], inputs[0], outputs[conv::kData]));
    conv_param.add_to = req[conv::kWeight] == kAddTo;
    ok                = ok && (req[conv::kWeight] == kNullOp ||
                cudnn::Exec<cudnn::ConvWgrad>(
                    ctx, conv_param, inputs[1 + conv::kData], inputs[0], outputs[conv::kWeight]));
    if (ok && !param.no_bias && req[conv::kBias] != kNullOp) {
      auto li     = cudnn::GetLayoutInfo(static_cast<mshadow::LayoutFlag>(param.layout.value()));
      auto add_to = req[conv::kBias] == kAddTo;
      if (li.channel_last ||
          !cudnn::LegacyBiasGrad(ctx, li, add_to, outputs[conv::kBias], inputs[0])) {
        if (li.channel_last) {
          // This kernel should be faster.
          auto y_grad = FlattenAs2DHead<gpu, DType>(inputs[0], ctx);
          AddBiasGrad(outputs[conv::kBias], y_grad, req[conv::kBias], param.num_filter, ctx);
        } else {
          TShape axes{static_cast<int>(li.ChannelIdx())};
          TShape small = ReduceAxesShapeImpl(
              inputs[0].shape_, dmlc::optional<mxnet::TShape>(axes), true, true);
          ReduceAxesRTCComputeImpl(
              ctx, {inputs[0]}, {req[conv::kBias]}, {outputs[conv::kBias]}, small, "red::sum{}");
        }
      }
    }
    if (!ok) {
      if (!param.cudnn_off)
        LOG(WARNING) << "This convolution backward is not supported by cuDNN, MXNet op is applied.";
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    }
  })
#else
  if (param.layout.value() != kNCW && param.layout.value() != kNCHW &&
      param.layout.value() != kNCDHW) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    })
    return;
  }

  if (param.num_filter == param.num_group && param.layout.value() == mshadow::kNCHW &&
      param.num_filter == in_data[conv::kData].shape_[1] && param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) && dtype == mshadow::kFloat32) {
    // The first element stores out grad.
    mxnet::ShapeVector in_shape(in_data.size());
    mxnet::ShapeVector out_shape(1, out_grad.shape_);
    for (size_t i = 0; i < in_shape.size(); i++)
      in_shape[i] = in_data[i].shape_;
    DepthwiseConvolutionOp<float> op;
    op.Init(param, in_shape, out_shape);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    return;
  }

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    ConvolutionOp<gpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  })
#endif  // MXNET_USE_CUDNN
}

NNVM_REGISTER_OP(Convolution).set_attr<FCompute>("FCompute<gpu>", ConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_Convolution)
    .set_attr<FCompute>("FCompute<gpu>", ConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
