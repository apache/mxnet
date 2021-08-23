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
 * \file cudnn_deconvolution-inl.h
 * \brief
 * \author Wei Wu, Leonard Lausen
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_

#include <mxnet/storage.h>
#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include "../deconvolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../../../common/cuda/utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

template<typename DType>
class CuDNNDeconvolutionOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);

 public:
  CuDNNDeconvolutionOp() {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&forward_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&back_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&back_conv_desc_w_));
  }

  void Init(DeconvolutionParam param,
            int forward_compute_type,
            int backward_compute_type,
            const mxnet::ShapeVector& in_shape,
            const mxnet::ShapeVector& out_shape,
            const RunContext& rctx,
            bool add_to_weight) {
    using namespace mshadow;
    this->param_ = param;
    this->add_to_weight_ = add_to_weight;
    InitBufferForParam();
    auto cudnn_forward_compute_type = convertToCuDNNDataType(forward_compute_type);
    auto cudnn_backward_compute_type = convertToCuDNNDataType(backward_compute_type);
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    // TensorCore algos only allowed on fp16-I/O deconvolutions if permitted by the global policy.
    cudnn_tensor_core_ = DataType<DType>::kFlag == kFloat16 && GetEnvAllowTensorCore();

    auto effective_layout = param_.layout.value();
    switch (effective_layout) {
      // 1D convolutions will be executed as 2D convolutions with a height of 1.
      case mshadow::kNCW: effective_layout = mshadow::kNCHW; break;
      case mshadow::kNWC: effective_layout = mshadow::kNHWC; break;
      case mshadow::kCWN: effective_layout = mshadow::kCHWN; break;
      default: break;
    }

    MSHADOW_LAYOUT_SWITCH(effective_layout, Layout, {
        format_ = LayoutType<Layout>::kCudnnFlag;
      });
    // Double check to make sure this class supports the operation
    if (!Supports(param, forward_compute_type, backward_compute_type, rctx.ctx.dev_id))
      LOG(FATAL) << "Deconvolution parameters not supported by cuDNN implementation.";

    InitDescriptors(in_shape, out_shape,
                    cudnn_forward_compute_type, cudnn_backward_compute_type);

    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }
    // In cuDNN_v6, dilated convolution descriptors are compatible with only a
    // single convolution algorithm.  Despite this, we go through the algorithm
    // selection process, which will return the only algorithm supported.  This
    // approach keeps the treatment of convolution cases uniform and will
    // naturally respond to more algorithms supporting dilated convolutions in
    // future cuDNN releases.
    SelectAlgo(rctx, in_shape, out_shape,
               cudnn_forward_compute_type, cudnn_backward_compute_type);
  }

  ~CuDNNDeconvolutionOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(forward_conv_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(back_conv_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(back_conv_desc_w_));
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    GetTempSize(ctx);
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, forward_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    DType *data_ptr = GetNdPtr(in_data[deconv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[deconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *out_ptr = GetNdPtr(out_data[deconv::kOut], param_.kernel.ndim() + 2, s);

    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta  = 0.0f;
      CUDNN_CALL(cudnnConvolutionBackwardData(s->dnn_handle_,
                 &alpha,
                 filter_desc_,
                 wmat_ptr + weight_offset_ * g,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 forward_conv_desc_,  // this backward algorithm used for inference
                 back_algo_.AlgoNumber(),
                 workspace.dptr_,
                 workspace_size,
                 &beta,
                 out_desc_,
                 out_ptr + out_offset_ * g));
      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[deconv::kBias].get<gpu, 1, DType>(s);
        CUDNN_CALL(cudnnAddTensor(s->dnn_handle_,
                                  &alpha,
                                  bias_desc_,
                                  bias.dptr_ + bias_offset_ * g,
                                  &beta,
                                  out_desc_,
                                  out_ptr + out_offset_ * g));
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), param_.no_bias ? 2U : 3U);
    CHECK_EQ(in_grad.size(), expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    // I/O's should have 2 more dims than the kernel dim
    DType *grad_ptr = GetNdPtr(out_grad[deconv::kOut], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[deconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *gwmat_ptr = GetNdPtr(in_grad[deconv::kWeight], param_.kernel.ndim() + 2, s);
    DType *data_ptr = GetNdPtr(in_data[deconv::kData], param_.kernel.ndim() + 2, s);
    DType *gdata_ptr = GetNdPtr(in_grad[deconv::kData], param_.kernel.ndim() + 2, s);

    CHECK_NE(req[deconv::kWeight], kWriteInplace);
    if (!param_.no_bias) {
      CHECK_NE(req[deconv::kBias], kWriteInplace);
    }
    CHECK_NE(req[deconv::kData], kWriteInplace);
    GetTempSize(ctx);
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, backward_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType bias_beta = 0.0f;
      if (!param_.no_bias && req[deconv::kBias] == kAddTo) {
        bias_beta = 1.0f;
      }
      typename DataType<DType>::ScaleType data_beta =
        req[deconv::kData] == kAddTo ? 1.0f : 0.0f;
      typename DataType<DType>::ScaleType weight_beta =
        req[deconv::kWeight] == kAddTo ? 1.0f : 0.0f;
      if (req[deconv::kWeight] != kNullOp) {
       CHECK_EQ(add_to_weight_, req[deconv::kWeight] == kAddTo);
        CUDNN_CALL(cudnnConvolutionBackwardFilter(
          s->dnn_handle_,
          &alpha,
          out_desc_,
          grad_ptr + out_offset_ * g,
          in_desc_,
          data_ptr + data_offset_ * g,
          back_conv_desc_,
          back_algo_w_.AlgoNumber(),
          workspace.dptr_,
          workspace_size,
          &weight_beta,
          filter_desc_,
          gwmat_ptr + weight_offset_ * g));
      }
      if (!param_.no_bias && (req[deconv::kBias] != kNullOp)) {
        Tensor<gpu, 1, DType> gbias = in_grad[deconv::kBias].get<gpu, 1, DType>(s);
        CUDNN_CALL(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                                &alpha,
                                                out_desc_,
                                                grad_ptr + out_offset_ * g,
                                                &bias_beta,
                                                bias_desc_,
                                                gbias.dptr_ + bias_offset_ * g));
      }
      if (req[deconv::kData] != kNullOp) {
        CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                                           &alpha,
                                           out_desc_,
                                           grad_ptr + out_offset_ * g,
                                           filter_desc_,
                                           wmat_ptr + weight_offset_ * g,
                                           back_conv_desc_,
                                           forward_algo_.AlgoNumber(),
                                           workspace.dptr_,
                                           workspace_size,
                                           &data_beta,
                                           in_desc_,
                                           gdata_ptr + data_offset_ * g));
      }
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the deconvolution
 * operation described by `param`: cuDNN v5 and earlier does not support
 * dilated convolutions.
 */
  static bool Supports(DeconvolutionParam param,
                       int forward_compute_type,
                       int backward_compute_type,
                       int dev_id) {
    using namespace mshadow;

    // NDHWC not supported, NHWC not supported in true fp16
    auto layout_val = param.layout.value();
    auto true_fp16 = DataType<DType>::kFlag == kFloat16 &&
      (forward_compute_type == kFloat16 || backward_compute_type == kFloat16);
    if (layout_val == kNDHWC || layout_val == kNWC ||
        layout_val == kNHWC && true_fp16)
      return false;

    // Permits graceful fallback to pseudo-fp16 on heterogenous systems
    if (!SupportsFloat16Compute(dev_id) &&
        (forward_compute_type == kFloat16 || backward_compute_type == kFloat16)) {
      return false;
    }

    // The factor by which the effective filter size grows based on dilation.
    auto filterDilationFactor = param.dilate.Size();

    return true;
  }

 private:
/*!
 * \brief Translate an mxnet datatype to the corresponding cudnnDataType_t.
 */
  cudnnDataType_t convertToCuDNNDataType(int dtype) {
    cudnnDataType_t converted = CUDNN_DATA_FLOAT;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_REAL_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

  inline void InitDescriptors(const mxnet::ShapeVector &in_shape,
                              const mxnet::ShapeVector &out_shape,
                              cudnnDataType_t cudnn_forward_compute_type,
                              cudnnDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);

    mxnet::TShape dshape = in_shape[deconv::kData];
    mxnet::TShape wshape = in_shape[deconv::kWeight];
    mxnet::TShape oshape = out_shape[deconv::kOut];
    mxnet::TShape dstride, ostride;
    wshape[0] /= param_.num_group;
    if (param_.kernel.ndim() == 1 || param_.kernel.ndim() == 2) {
      // 1d or 2d conv
      index_t o_pad[2];
      index_t o_adj[2];
      if (param_.kernel.ndim() == 2) {
        param_.InferPad(dshape, o_pad, o_adj);
      } else {
        index_t o_pad_1D[1];
        index_t o_adj_1D[1];
        param_.InferPad(dshape, o_pad_1D, o_adj_1D);
        o_pad[0] = 0;
        o_pad[1] = o_pad_1D[0];
      }
      auto stride = param_.kernel.ndim() == 2 ?
        param_.stride : mxnet::TShape({1, param_.stride[0]});
      auto dilate = param_.kernel.ndim() == 2 ?
        param_.dilate : mxnet::TShape({1, param_.dilate[0]});

      CUDNN_CALL(cudnnSetConvolution2dDescriptor(forward_conv_desc_,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilate[0],
                                                 dilate[1],
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_forward_compute_type));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(back_conv_desc_,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilate[0],
                                                 dilate[1],
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(back_conv_desc_w_,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilate[0],
                                                 dilate[1],
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));
      if (param_.kernel.ndim() == 2) {
        wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
        dstride = ConvertLayout(Strides<4>(dshape), param_.layout.value(), kNCHW);
        dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        ostride = ConvertLayout(Strides<4>(oshape), param_.layout.value(), kNCHW);
        oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
      } else {
        wshape = ConvertLayout(wshape.get<3>(), param_.layout.value(), kNCW);
        wshape = mxnet::TShape({wshape[0], wshape[1], 1, wshape[2]});
        dstride = ConvertLayout(Strides<3>(dshape), param_.layout.value(), kNCW);
        dstride = mxnet::TShape({dstride[0], dstride[1], dstride[1], dstride[2]});
        dshape = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
        dshape = mxnet::TShape({dshape[0], dshape[1], 1, dshape[2]});
        ostride = ConvertLayout(Strides<3>(oshape), param_.layout.value(), kNCW);
        ostride = mxnet::TShape({ostride[0], ostride[1], ostride[1], ostride[2]});
        oshape = ConvertLayout(oshape.get<3>(), param_.layout.value(), kNCW);
        oshape = mxnet::TShape({oshape[0], oshape[1], 1, oshape[2]});
      }
      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                            dtype_,
                                            format_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));
#if CUDNN_VERSION >= 7301 && CUDNN_VERSION < 7500
      auto kernel_h = wshape[2];
      auto kernel_w = wshape[3];
      auto stride_h = stride[0];
      auto stride_w = stride[1];
      auto pad_h = o_pad[0];
      auto pad_w = o_pad[1];
      if (param_.layout.value() == kNCHW &&
          (((stride_h == 2) && (kernel_h % 2 == 0) && (pad_h % 2 == 0)) ||
           ((stride_w == 2) && (kernel_w % 2 == 0) && (pad_w % 2 == 0)))) {
        exclude_dgrad_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
      }
#endif
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      index_t o_pad[3];
      index_t o_adj[3];
      param_.InferPad(dshape, o_pad, o_adj);

      CHECK_EQ(param_.layout.value(), kNCDHW) << "CuDNN only support 3D conv with NCDHW layout";
      std::vector<int> wshape_buffer(wshape.ndim());
      CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc_,
                                            dtype_,
                                            CUDNN_TENSOR_NCHW,
                                            static_cast<int>(wshape.ndim()),
                                            CastTShapeToIntPtr(wshape, &wshape_buffer)));
      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(forward_conv_desc_,
                                                 3,
                                                 reinterpret_cast<int*>(&o_pad[0]),
                                                 param_stride_.data(),
                                                 param_dilate_.data(),
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_forward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(back_conv_desc_,
                                                 3,
                                                 reinterpret_cast<int*>(&o_pad[0]),
                                                 param_stride_.data(),
                                                 param_dilate_.data(),
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(back_conv_desc_w_,
                                                 3,
                                                 reinterpret_cast<int*>(&o_pad[0]),
                                                 param_stride_.data(),
                                                 param_dilate_.data(),
                                                 CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));

      dstride = ConvertLayout(Strides<5>(dshape), param_.layout.value(), kNCDHW);
      dshape = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
      ostride = ConvertLayout(Strides<5>(oshape), param_.layout.value(), kNCDHW);
      oshape = ConvertLayout(oshape.get<5>(), param_.layout.value(), kNCDHW);
    }
    // Set "allow tensor core" flag in convolution descriptors, if available.
    cudnnMathType_t math_type = cudnn_tensor_core_ ? CUDNN_TENSOR_OP_MATH
                                                   : CUDNN_DEFAULT_MATH;
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w_, math_type));
    dshape[1] /= param_.num_group;
    oshape[1] /= param_.num_group;
    weight_offset_ = wshape.Size();
    data_offset_ = dstride[1] * dshape[1];
    out_offset_ = ostride[1] * oshape[1];

    std::vector<int> dshape_buffer(dshape.ndim());
    std::vector<int> dstride_buffer(dstride.ndim());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                          dtype_,
                                          static_cast<int>(dshape.ndim()),
                                          CastTShapeToIntPtr(dshape, &dshape_buffer),
                                          CastTShapeToIntPtr(dstride, &dstride_buffer)))

    std::vector<int> oshape_buffer(oshape.ndim());
    std::vector<int> ostride_buffer(ostride.ndim());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
                                          static_cast<int>(oshape.ndim()),
                                          CastTShapeToIntPtr(oshape, &oshape_buffer),
                                          CastTShapeToIntPtr(ostride, &ostride_buffer)));

    if (!param_.no_bias) {
      mxnet::TShape bias = in_shape[deconv::kBias];
      bias_offset_ = bias[0] / param_.num_group;
      int bias_dim = static_cast<int>(bias_offset_);
      std::vector<int> bias_shape = {1,
                                     bias_dim,
                                     1, 1};
      std::vector<int> bias_stride = {bias_dim, 1, bias_dim, bias_dim};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(bias_dim);
      }
      CUDNN_CALL(cudnnSetTensorNdDescriptor(bias_desc_,
                                            dtype_,
                                            static_cast<int>(bias_shape.size()),
                                            &bias_shape[0],
                                            &bias_stride[0]));
    }
  }

  void CuDNNAlgoSetter(const RunContext& rctx,
                       const mxnet::ShapeVector& in_shape,
                       const mxnet::ShapeVector& out_shape,
                       cudnnDataType_t cudnn_forward_compute_type,
                       cudnnDataType_t cudnn_backward_compute_type,
                       CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
                       CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
                       CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
    // Not in algo registry, must determine via *Get*() or *Find*()
    mshadow::Stream <gpu> *s = rctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));

    // Since the function signature of *Get*_v7() matches that of *Find*(),
    // we can unify the find-vs-get logic by using function pointers.

    // Forward Algorithm Find/Get() v7
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_results(MaxForwardAlgos(s->dnn_handle_));
    int actual_fwd_algos = 0;
    auto fwd_algo_discoverer =
        param_.cudnn_tune.value() == deconv::kOff ? cudnnGetConvolutionForwardAlgorithm_v7
                                                : cudnnFindConvolutionForwardAlgorithm;
    CUDNN_CALL((*fwd_algo_discoverer)(s->dnn_handle_,
                                      out_desc_,
                                      filter_desc_,
                                      back_conv_desc_,  // fwd algo used to backprop-to-data
                                      in_desc_,
                                      fwd_results.size(),
                                      &actual_fwd_algos,
                                      fwd_results.data()));
    fwd_results.resize(actual_fwd_algos);
    AlgoFinalSelect<cudnnConvolutionFwdAlgoPerf_t,
        cudnnConvolutionFwdAlgo_t>(fwd_results, "forward",
                                   workspace_byte, fwd);

    // Backprop-to-Filter Algorithm Find/Get() v7
    auto max_bwd_filt_algos = MaxBackwardFilterAlgos(s->dnn_handle_);
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filt_results(max_bwd_filt_algos);
    int actual_bwd_filter_algos = 0;
    // In cudnn v7.1.4, find() returned wgrad algos that could fail for large c if we
    // were summing into the output (i.e. beta != 0).  Get() returned OK algos though.
    auto bwd_filter_algo_discoverer =
        param_.cudnn_tune.value() == deconv::kOff ? cudnnGetConvolutionBackwardFilterAlgorithm_v7
                                                  : cudnnFindConvolutionBackwardFilterAlgorithm;
    CUDNN_CALL((*bwd_filter_algo_discoverer)(s->dnn_handle_,
                                             out_desc_,
                                             in_desc_,
                                             back_conv_desc_,
                                             filter_desc_,
                                             bwd_filt_results.size(),
                                             &actual_bwd_filter_algos,
                                             bwd_filt_results.data()));
    bwd_filt_results.resize(actual_bwd_filter_algos);
    AlgoFinalSelect<cudnnConvolutionBwdFilterAlgoPerf_t,
        cudnnConvolutionBwdFilterAlgo_t>(bwd_filt_results, "backprop-to-filter",
                                         workspace_byte, flt);
    // Backprop-to-Data Algorithm Find/Get() v7
    auto max_bwd_data_algos = MaxBackwardDataAlgos(s->dnn_handle_);
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_results(max_bwd_data_algos);
    int actual_bwd_data_algos = 0;
    auto bwd_data_algo_discoverer =
        param_.cudnn_tune.value() == deconv::kOff ? cudnnGetConvolutionBackwardDataAlgorithm_v7
                                                : cudnnFindConvolutionBackwardDataAlgorithm;
    CUDNN_CALL((*bwd_data_algo_discoverer)(s->dnn_handle_,
                                           filter_desc_,
                                           in_desc_,
                                           forward_conv_desc_,  // bwd algo used in inference
                                           out_desc_,
                                           bwd_data_results.size(),
                                           &actual_bwd_data_algos,
                                           bwd_data_results.data()));
    bwd_data_results.resize(actual_bwd_data_algos);
    AlgoFinalSelect<cudnnConvolutionBwdDataAlgoPerf_t,
        cudnnConvolutionBwdDataAlgo_t>(bwd_data_results, "backprop-to-data",
                                       workspace_byte, bwd, exclude_dgrad_algo_);

    // Fix for issue #11241
    int cudnn_find_issue_max_features = 64 * 1024;
    // With deconvolution, the algo sensitivity is to a large number of output features
    if (add_to_weight_ && Features(out_shape[deconv::kOut]) >= cudnn_find_issue_max_features) {
      flt->Set(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, true);
    }
  }

  void SelectAlgo(const RunContext& rctx,
                  const mxnet::ShapeVector& in_shape,
                  const mxnet::ShapeVector& out_shape,
                  cudnnDataType_t cudnn_forward_compute_type,
                  cudnnDataType_t cudnn_backward_compute_type) {
    auto algo_setter = [&](CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
                           CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
                           CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
      if (param_.cudnn_tune.value() == deconv::kOff) {
        // The routine will only be calling cudnnGet, so no need to grab the Storage lock.
        this->CuDNNAlgoSetter(rctx, in_shape, out_shape,
                              cudnn_forward_compute_type,
                              cudnn_backward_compute_type,
                              fwd, bwd, flt);
      } else {
        // One potential problem is that cudnnFind() uses cudaMalloc() to directly allocate
        // I/O and workspace areas, and these allocations may result in an out-of-memory
        // error even though the StorageMangager free pool is not empty.  Ideally, cudnnFind
        // would use MXNet's storage allocator for its I/O and workspace areas, instead of using
        // the area carved out by MXNET_GPU_MEM_POOL_RESERVE.
        // To get somewhat the same effect as this, we can pre-allocate the areas needed for the
        // I/Os (possibly triggering a desirable StorageManager::ReleaseAll()), followed by a
        // DirectFree(), which makes these areas available for cudnn's subsequent cudaMalloc().

        // Allocate for x (or dx), w (or dw) and y (or dy).
        ReserveElements({in_shape[deconv::kData].Size(),
                         in_shape[deconv::kWeight].Size(),
                         out_shape[deconv::kOut].Size()});

        // We're about to call cudnnFind so we need to quiet the system by grabbing
        // the Storage lock.  Concurrent cudaMalloc's can disrupt the accurate timing
        // measurements of the algos, and can prevent the cuda driver's proper freeing
        // of cudnnFind's internal temporary allocations.  Grabbing the lock might also
        // impede other threads from launching work on the GPU.
        std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
        this->CuDNNAlgoSetter(rctx, in_shape, out_shape,
                              cudnn_forward_compute_type,
                              cudnn_backward_compute_type,
                              fwd, bwd, flt);
      }
    };

    // An algo specification by the user may be cached here, but another
    // convolution will match only if identically specified.
    // We're caching results of *Get* as well as *Find*, but these records
    // will be held distinctly because param_.cudnn_tune is part of the key.
    CuDNNDeconvAlgoReg::Get()->FindOrElseRegister(param_, in_shape, out_shape, dtype_,
                                         cudnn_forward_compute_type,
                                         cudnn_backward_compute_type,
                                         SMArch(rctx.ctx.dev_id), add_to_weight_,
                                         &forward_algo_, &back_algo_, &back_algo_w_, algo_setter);

    // If we're allowing Tensor Core variants of the algos to be considered in
    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.

    // The next two code lines will look like they have typos, but they don't!
    // The forward_conv_desc_ is used during inference, which invokes the back_algo_.
    // Thus, the mathType of the back_algo_ should be stored in the forward_conv_desc_.
    // Conversely, the back_conv_desc_ is used during training backprop, which invokes
    // the forward_algo_.  Thus, the mathType of the forward_algo_ should be stored
    // in the back_conv_desc_.
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, back_algo_.MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_, forward_algo_.MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w_, back_algo_w_.MathType()));
  }

  // Look over the results from *Find*() or *Get*() and pick the fastest algo given possible
  // workspace constraints and a possible user algo preference.
  template <typename PerfType, typename AlgoType>
  void AlgoFinalSelect(const std::vector<PerfType> &perf_results, std::string kernel_name,
                       size_t workspace_byte, CuDNNAlgo<AlgoType> *algo,
                       int32_t algo_exclude = -1) {
    // Determine the fastest acceptable algo regardless of mathType.
    bool enforce_determinism = dmlc::GetEnv("MXNET_ENFORCE_DETERMINISM", false);
    for (decltype(perf_results.size()) i = 0; i != perf_results.size(); ++i) {
      const auto &result = perf_results[i];
      bool algo_exclusion = static_cast<int32_t>(result.algo) == algo_exclude;
      bool algo_is_tensor_core = false;
      algo_is_tensor_core = result.mathType == CUDNN_TENSOR_OP_MATH;
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (!enforce_determinism || result.determinism == cudnnDeterminism_t::CUDNN_DETERMINISTIC) &&
          (param_.cudnn_tune.value() != deconv::kLimited || result.memory <= workspace_byte) &&
          !algo_exclusion) {
        algo->Set(result.algo, algo_is_tensor_core);
        return;
      }
    }
    auto mode = param_.cudnn_tune.value() == deconv::kOff ? " get " : " find ";
    LOG(FATAL) << "Failed to" << mode << "any " << kernel_name << " deconvolution algorithm"
               << " with workspace size of " << workspace_byte << " bytes,"
               << " please consider reducing batch/model size or increasing the workspace size";
  }


  void GetTempSize(const OpContext& ctx) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t back_data_algo_workspace_size = 0;
    size_t back_filter_algo_workspace_size = 0;
    size_t forward_algo_workspace_size = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               in_desc_,
               forward_conv_desc_,
               out_desc_,
               back_algo_.AlgoNumber(),
               &back_data_algo_workspace_size));
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               out_desc_,
               in_desc_,
               back_conv_desc_,
               filter_desc_,
               back_algo_w_.AlgoNumber(),
               &back_filter_algo_workspace_size));
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               out_desc_,
               filter_desc_,
               back_conv_desc_,
               in_desc_,
               forward_algo_.AlgoNumber(),
               &forward_algo_workspace_size));

    forward_workspace_byte_ = back_data_algo_workspace_size;
    backward_workspace_byte_ = std::max(forward_algo_workspace_size,
                                        back_filter_algo_workspace_size);
  }

  int *CastTShapeToIntPtr(const mxnet::TShape& s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = nullptr;
    if (dim == 3) {
      Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

  // Converts a mxnet::TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const mxnet::TShape &s) {
    int ndim = s.ndim();
    mxnet::TShape strides(ndim, -1);
    for (int i = 0; i != ndim; ++i)
      strides[i] = s.ProdShape(i+1, ndim);
    return strides.get<dim>();
  }

  void InitBufferForParam() {
    CastTShapeToIntPtr(param_.stride, &param_stride_);
    CastTShapeToIntPtr(param_.dilate, &param_dilate_);
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx, size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words = size_bytes / sizeof(DType) + 1;
    return ctx.requested[deconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(size_words), s);
  }

  // Returns the size in bytes of the 1D Tensor of words.
  size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, DType> &tensor) {
    return tensor.MSize() * sizeof(DType);
  }


  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const mxnet::TShape &dshape) {
    int c = 0;
    switch (dshape.ndim()) {
      case 3: c = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW)[1]; break;
      case 4: c = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW)[1]; break;
      case 5: c = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW)[1]; break;
      default:
        LOG(FATAL) << "Unexpected deconvolution data dimension " << dshape.ndim();
    }
    return c;
  }

  // Make a number of allocations and directly free them, ensuring room for an equivalent set of
  // cudaMalloc() calls by (say) cudnnFind().  `elements` spec the alloc size in DTypes, not bytes.
  void ReserveElements(const std::vector<size_t> &elements) {
    std::vector<Storage::Handle> handles;
    for (size_t alloc_element : elements) {
        handles.push_back(Storage::Get()->Alloc(alloc_element * sizeof(DType), Context::GPU()));
        handles.back().profiler_scope = "<ephemeral>:";
        handles.back().name = "reserve_elements";
    }
    for (auto &handle : handles)
        Storage::Get()->DirectFree(handle);
  }


  // Log that no suitable algo was found that met the workspace constraints, then exit.
  void LogNoSuitableAlgoAndExit(int num_algos_tried, size_t min_memory_needs,
                                size_t workspace_byte, std::string algo_kind) {
    LOG(FATAL) << num_algos_tried << " " << algo_kind << " with minimum memory requirement "
               << min_memory_needs << " bytes have been tried. Workspace size is set to "
               << workspace_byte << " bytes, please consider reducing the batch/model size, "
               << "or increasing workspace size.";
  }

  std::vector<int> param_stride_;
  std::vector<int> param_dilate_;

  int forward_compute_type_;
  int backward_compute_type_;
  const mxnet::ShapeVector in_shapes_;
  const mxnet::ShapeVector out_shapes_;

  // Temp workspace size in bytes needed for Forward() operation.  Note that
  // in deconvolution, this is handled by the cuDNN backprop-to-data kernel.
  size_t forward_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() operation.  Note that
  // in deconvolution, this is handled by the cuDNN forward kernel and the
  // the cuDNN backprop-to-filter kernel.
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  // Convolution descriptor for "forward" inference operation.
  // Note that in deconvolution, the forward operation is handled
  // by the cuDNN backprop-to-data kernel.
  cudnnConvolutionDescriptor_t forward_conv_desc_;
  // Convolution descriptor for "back-prop" operations to data .
  // Note that in deconvolution, the backprop-to-data operation is handled
  // by the cuDNN forward kernel.
  cudnnConvolutionDescriptor_t back_conv_desc_;
  // Convolution descriptor for "back-prop" operations to filter.
  // Note that in deconvolution, the backprop-to-data operation is handled
  // by the backprop-to-filter kernel (so consistent with the treatment
  // in convolution).
  cudnnConvolutionDescriptor_t back_conv_desc_w_;
  // Algorithm for the cuDNN forward kernel (used in gradient backprop to input)
  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> forward_algo_;
  // Algorithm for the cuDNN backprop-to-data kernel (used in inference)
  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> back_algo_;
  // Algorithm for the cuDNN backprop-to-filter kernel
  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> back_algo_w_;
  cudnnTensorFormat_t format_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;
  // Is req[kWeight] == deconv::kAddTo ?
  bool add_to_weight_;
  // Is there a dgrad algo that should be avoided (-1 == none)?
  int32_t exclude_dgrad_algo_ = -1;
  DeconvolutionParam param_;
};
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_
