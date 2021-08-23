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
 * Copyright (c) 2017 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
*/

#include "./convolution-inl.h"
#include <vector>
#include "./depthwise_convolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNConvolutionOp<DType>& GetCuDNNConvOp(const ConvolutionParam& param,
                                                 int forward_compute_type,
                                                 int backward_compute_type,
                                                 const mxnet::ShapeVector& in_shape,
                                                 const mxnet::ShapeVector& out_shape,
                                                 const RunContext& rctx,
                                                 bool add_to_weight) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<ConvSignature,
                                         std::shared_ptr<CuDNNConvolutionOp<DType> >,
                                         OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<ConvSignature,
                                            std::shared_ptr<CuDNNConvolutionOp<DType> >,
                                            OpHash> ops;
#endif
  ConvSignature key(param);
  size_t ndim = 0;
  for (auto &s : in_shape)
    ndim += s.ndim();
  for (auto &s : out_shape)
    ndim += s.ndim();
  key.Reserve(1 /* for forward_compute_type */ +
              1 /* for backward_compute_type */ +
              ndim /* for in and out shapes */ +
              1 /* for dev_id */ +
              1 /* for add_to_weight */);

  key.AddSign(forward_compute_type);
  key.AddSign(backward_compute_type);
  key.AddSign(in_shape);
  key.AddSign(out_shape);
  key.AddSign(rctx.ctx.dev_id);
  key.AddSign(add_to_weight ? 1 : 0);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuDNNConvolutionOp<DType>> op(new CuDNNConvolutionOp<DType>());
    auto ins_ret = ops.insert(std::pair<ConvSignature, std::shared_ptr<CuDNNConvolutionOp<DType>>>(
                              key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(param, forward_compute_type, backward_compute_type, in_shape,
                     out_shape, rctx, add_to_weight);
  }
  return *it->second;
}
#endif

template<>
void ConvolutionCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  int dtype = inputs[conv::kData].type_flag_;

#if MXNET_USE_CUDNN == 1
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);
  // On fp16-I/O instances, use fp32 compute (i.e. pseudo-fp16).
  int compute_type = (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    } else if (!CuDNNConvolutionOp<DType>::Supports(param,
          compute_type, compute_type, ctx.run_ctx.ctx.dev_id)) {
      LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    } else {
      mxnet::ShapeVector in_shape(inputs.size());
      mxnet::ShapeVector out_shape(1, outputs[0].shape_);
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = inputs[i].shape_;
      // req[conv::kWeight] is only set for backward, so assume the typical 'write' for now.
      auto add_to_weight = false;
      CuDNNConvolutionOp<DType> &op = GetCuDNNConvOp<DType>(param,
          compute_type, compute_type, in_shape, out_shape, ctx.run_ctx, add_to_weight);
      op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == inputs[conv::kData].shape_[1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
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

template<>
void ConvolutionGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;
  int dtype = out_grad.type_flag_;

#if MXNET_USE_CUDNN == 1
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);
  // On fp16-I/O instances, use fp32 compute (i.e. pseudo-fp16).
  int compute_type = (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    } else if (!CuDNNConvolutionOp<DType>::Supports(param,
          compute_type, compute_type, ctx.run_ctx.ctx.dev_id)) {
      LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
      ConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    } else {
      // The first element stores out grad.
      mxnet::ShapeVector in_shape(in_data.size());
      mxnet::ShapeVector out_shape(1, out_grad.shape_);
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = in_data[i].shape_;
      auto add_to_weight = req[conv::kWeight] == kAddTo;
      CuDNNConvolutionOp<DType> &op = GetCuDNNConvOp<DType>(param,
          compute_type, compute_type, in_shape, out_shape, ctx.run_ctx, add_to_weight);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    }
  })
#else
  if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == in_data[conv::kData].shape_[1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
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

NNVM_REGISTER_OP(Convolution)
.set_attr<FCompute>("FCompute<gpu>", ConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_Convolution)
.set_attr<FCompute>("FCompute<gpu>", ConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

