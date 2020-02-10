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
 * \file cudnn_convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_CONVOLUTION_INL_H_

#include <mxnet/storage.h>
#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include "../convolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../../../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

/*!
 * \brief The Operator used to perform convolution using cuDNN kernels.
 */
template<typename DType>
class CuDNNConvolutionOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);

 public:
  CuDNNConvolutionOp() {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&forward_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&back_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&back_conv_desc_w_));
    parallelize_backward_kernels_ = Context::GetGPUStreamsPerWorker() >= 2;
  }

  void Init(const ConvolutionParam& param,
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
    dtype_ = DataType<DType>::kCudnnFlag;
    // TensorCore algos only allowed on fp16-I/O convolutions if permitted by the global policy.
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
      LOG(FATAL) << "Convolution parameters not supported by cuDNN implementation.";

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
    GetTempSize(rctx);
  }

  ~CuDNNConvolutionOp() {
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
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, forward_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    DType *data_ptr = GetNdPtr(in_data[conv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[conv::kWeight], param_.kernel.ndim() + 2, s);
    DType *out_ptr = GetNdPtr(out_data[conv::kOut], param_.kernel.ndim() + 2, s);

    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    typename DataType<DType>::ScaleType beta_add = 1.0f;
    CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                    &alpha,
                    in_desc_,
                    data_ptr,
                    filter_desc_,
                    wmat_ptr,
                    forward_conv_desc_,
                    forward_algo_.AlgoNumber(),
                    workspace.dptr_,
                    workspace_size,
                    req[conv::kOut] == kAddTo? &beta_add : &beta,
                    out_desc_,
                      out_ptr));

    if (!param_.no_bias) {
      Tensor<gpu, 1, DType> bias = in_data[conv::kBias].get<gpu, 1, DType>(s);
      CUDNN_CALL(cudnnAddTensor(s->dnn_handle_,
                              &alpha,
                              bias_desc_,
                              bias.dptr_,
                              &beta_add,
                              out_desc_,
                              out_ptr));
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
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // RAII object to handle syncing of the underlying auxiliary stream with the primary stream
    SyncedGPUAuxStream s_dgrad = ctx.get_gpu_aux_stream();

    // I/O's should have 2 more dims than the kernel dim
    DType *grad_ptr = GetNdPtr(out_grad[conv::kOut], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[conv::kWeight], param_.kernel.ndim() + 2, s);
    DType *gwmat_ptr = GetNdPtr(in_grad[conv::kWeight], param_.kernel.ndim() + 2, s);
    DType *data_ptr = GetNdPtr(in_data[conv::kData], param_.kernel.ndim() + 2, s);
    DType *gdata_ptr = GetNdPtr(in_grad[conv::kData], param_.kernel.ndim() + 2, s);

    size_t backward_workspace_byte =
        parallelize_backward_kernels_ ? back_workspace_byte_dgrad_ + back_workspace_byte_wgrad_
                                      : std::max(back_workspace_byte_dgrad_,
                                                 back_workspace_byte_wgrad_);
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, backward_workspace_byte);
    size_t workspace_size = TensorSizeBytes(workspace);
    DType *workspace_dptr_wgrad = workspace.dptr_;
    DType *workspace_dptr_dgrad = workspace.dptr_;
    if (parallelize_backward_kernels_) {
      CHECK_LE(back_workspace_byte_dgrad_ + back_workspace_byte_wgrad_, workspace_size);
      // Large allocations at some point will be given their own page.  Pass this alignment on to
      // the larger of the two separate dgrad/wgrad workspaces.  This probably doesn't matter, but
      // corresponds more closely to the workspace alignments used during cudnnFind.
      if (back_workspace_byte_dgrad_ > back_workspace_byte_wgrad_)
        workspace_dptr_wgrad = workspace.dptr_ + back_workspace_byte_dgrad_ / sizeof(DType);
      else
        workspace_dptr_dgrad = workspace.dptr_ + back_workspace_byte_wgrad_ / sizeof(DType);
    } else {
      CHECK_LE(back_workspace_byte_dgrad_, workspace_size);
      CHECK_LE(back_workspace_byte_wgrad_, workspace_size);
    }
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    typename DataType<DType>::ScaleType beta_add = 1.0f;
    if (!param_.no_bias && (req[conv::kBias] != kNullOp)) {
        Tensor<gpu, 1, DType> gbias = in_grad[conv::kBias].get<gpu, 1, DType>(s);
        CUDNN_CALL(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                            &alpha,
                                            out_desc_,
                                            grad_ptr,
                                            req[conv::kBias] == kAddTo ? &beta_add : &beta,
                                            bias_desc_,
                                            gbias.dptr_));
    }
    if (req[conv::kWeight] != kNullOp) {
        CHECK_EQ(add_to_weight_, req[conv::kWeight] == kAddTo);
        CUDNN_CALL(cudnnConvolutionBackwardFilter(s->dnn_handle_,
            &alpha,
            in_desc_,
            data_ptr,
            out_desc_,
            grad_ptr,
            back_conv_desc_w_,
            back_algo_w_.AlgoNumber(),
            workspace_dptr_wgrad,
            back_workspace_byte_wgrad_,
            req[conv::kWeight] == kAddTo? &beta_add : &beta,
            filter_desc_,
            gwmat_ptr));
    }
    if (req[conv::kData] != kNullOp) {
        CUDNN_CALL(cudnnConvolutionBackwardData(s_dgrad.GetStream()->dnn_handle_,
            &alpha,
            filter_desc_,
            wmat_ptr,
            out_desc_,
            grad_ptr,
            back_conv_desc_,
            back_algo_.AlgoNumber(),
            workspace_dptr_dgrad,
            back_workspace_byte_dgrad_,
            req[conv::kData] == kAddTo? &beta_add : &beta,
            in_desc_,
            gdata_ptr));
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the convolution
 * operation described by `param`: cuDNN v5 and earlier does not support
 * dilated convolutions.  Dilation only enabled after v6.0.20.
 */
  static bool Supports(ConvolutionParam param,
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

  void InitDescriptors(const mxnet::ShapeVector& in_shape,
                       const mxnet::ShapeVector& out_shape,
                       cudnnDataType_t cudnn_forward_compute_type,
                       cudnnDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);

    mxnet::TShape dshape = in_shape[conv::kData];
    mxnet::TShape wshape = in_shape[conv::kWeight];
    mxnet::TShape oshape = out_shape[conv::kOut];
    mxnet::TShape dstride, ostride;

    if (param_.kernel.ndim() == 1 || param_.kernel.ndim() == 2) {
      // 1d or 2d conv
      auto pad = param_.kernel.ndim() == 2 ?
        param_.pad : mxnet::TShape({0, param_.pad[0]});
      auto stride = param_.kernel.ndim() == 2 ?
        param_.stride : mxnet::TShape({1, param_.stride[0]});
      auto dilate = param_.kernel.ndim() == 2 ?
        param_.dilate : mxnet::TShape({1, param_.dilate[0]});
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(forward_conv_desc_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_forward_compute_type));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(back_conv_desc_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_backward_compute_type));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(back_conv_desc_w_,
                                               pad[0],
                                               pad[1],
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
      auto pad_h = pad[0];
      auto pad_w = pad[1];
      if (param_.layout.value() == kNCHW &&
          (((stride_h == 2) && (kernel_h % 2 == 0) && (pad_h % 2 == 0)) ||
           ((stride_w == 2) && (kernel_w % 2 == 0) && (pad_w % 2 == 0)))) {
        exclude_dgrad_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
      }
#endif
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      CHECK_EQ(param_.layout.value(), kNCDHW) << "CuDNN only support 3D conv with NCDHW layout";
      std::vector<int> wshape_buffer(wshape.ndim());
      CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc_,
                                          dtype_,
                                          CUDNN_TENSOR_NCHW,
                                          static_cast<int>(wshape.ndim()),
                                          CastTShapeToIntPtr(wshape, &wshape_buffer)));
      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(forward_conv_desc_,
                                               3,
                                               param_pad_.data(),
                                               param_stride_.data(),
                                               param_dilate_.data(),
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_forward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(back_conv_desc_,
                                               3,
                                               param_pad_.data(),
                                               param_stride_.data(),
                                               param_dilate_.data(),
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_backward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(back_conv_desc_w_,
                                               3,
                                               param_pad_.data(),
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
#if CUDNN_VERSION >= 7200
    if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
        (DataType<DType>::kFlag != kFloat16))
      math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#endif
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w_, math_type));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(forward_conv_desc_, param_.num_group));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(back_conv_desc_, param_.num_group));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(back_conv_desc_w_, param_.num_group));

    std::vector<int> dshape_buffer(dshape.ndim());
    nnvm::ShapeTypeCast(dshape.begin(), dshape.end(), dshape_buffer.data());
    std::vector<int> dstride_buffer(dstride.ndim());
    nnvm::ShapeTypeCast(dstride.begin(), dstride.end(), dstride_buffer.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                          dtype_,
                                          static_cast<int>(dshape.ndim()),
                                          dshape_buffer.data(),
                                          dstride_buffer.data()));

    std::vector<int> oshape_buffer(oshape.ndim());
    nnvm::ShapeTypeCast(oshape.begin(), oshape.end(), oshape_buffer.data());
    std::vector<int> ostride_buffer(ostride.ndim());
    nnvm::ShapeTypeCast(ostride.begin(), ostride.end(), ostride_buffer.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
                                          static_cast<int>(oshape.ndim()),
                                          oshape_buffer.data(),
                                          ostride_buffer.data()));

    if (!param_.no_bias) {
      mxnet::TShape bias = in_shape[conv::kBias];
      std::vector<int> bias_shape = {1,
                                     static_cast<int>(bias[0]),
                                     1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias[0]), 1, 1, 1};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(1);
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
    mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));

    // Since the function signature of *Get*_v7() matches that of *Find*(),
    // we can unify the find-vs-get logic by using function pointers.

    // Forward Algorithm Find/Get() v7
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_results(MaxForwardAlgos(s->dnn_handle_));
    int actual_fwd_algos = 0;
    auto fwd_algo_discoverer =
      param_.cudnn_tune.value() == conv::kOff ? cudnnGetConvolutionForwardAlgorithm_v7
                                              : cudnnFindConvolutionForwardAlgorithm;
    CUDNN_CALL((*fwd_algo_discoverer)(s->dnn_handle_,
                                      in_desc_,
                                      filter_desc_,
                                      forward_conv_desc_,
                                      out_desc_,
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
      param_.cudnn_tune.value() == conv::kOff ? cudnnGetConvolutionBackwardFilterAlgorithm_v7
                                              : cudnnFindConvolutionBackwardFilterAlgorithm;
    CUDNN_CALL((*bwd_filter_algo_discoverer)(s->dnn_handle_,
                                             in_desc_,
                                             out_desc_,
                                             back_conv_desc_w_,
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
      param_.cudnn_tune.value() == conv::kOff ? cudnnGetConvolutionBackwardDataAlgorithm_v7
                                              : cudnnFindConvolutionBackwardDataAlgorithm;
    CUDNN_CALL((*bwd_data_algo_discoverer)(s->dnn_handle_,
                                           filter_desc_,
                                           out_desc_,
                                           back_conv_desc_,
                                           in_desc_,
                                           bwd_data_results.size(),
                                           &actual_bwd_data_algos,
                                           bwd_data_results.data()));
    bwd_data_results.resize(actual_bwd_data_algos);
    AlgoFinalSelect<cudnnConvolutionBwdDataAlgoPerf_t,
                    cudnnConvolutionBwdDataAlgo_t>(bwd_data_results, "backprop-to-data",
                                                   workspace_byte, bwd, exclude_dgrad_algo_);

    // Fix for issue #11241
    int cudnn_find_issue_max_features = 64 * 1024;
    if (add_to_weight_ && Features(in_shape[conv::kData]) >= cudnn_find_issue_max_features) {
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
      if (param_.cudnn_tune.value() == conv::kOff) {
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
        ReserveElements({in_shape[conv::kData].Size(),
                         in_shape[conv::kWeight].Size(),
                         out_shape[conv::kOut].Size()});

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

    CuDNNConvAlgoReg::Get()->FindOrElseRegister(param_, in_shape, out_shape, dtype_,
                                       cudnn_forward_compute_type,
                                       cudnn_backward_compute_type,
                                       SMArch(rctx.ctx.dev_id), add_to_weight_,
                                       &forward_algo_, &back_algo_, &back_algo_w_, algo_setter);

    // If we're allowing Tensor Core variants of the algos to be considered in
    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, forward_algo_.MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_, back_algo_.MathType()));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w_, back_algo_w_.MathType()));
  }

  // Look over the results from *Find*() or *Get*() and pick the fastest algo given possible
  // workspace constraints.
  template <typename PerfType, typename AlgoType>
  void AlgoFinalSelect(const std::vector<PerfType> &perf_results, std::string kernel_name,
                       size_t workspace_byte, CuDNNAlgo<AlgoType> *algo,
                       int32_t algo_exclude = -1) {
    // Determine the fastest acceptable algo that matches the algo_preference (-1 = any),
    // regardless of mathType.
    bool enforce_determinism = dmlc::GetEnv("MXNET_ENFORCE_DETERMINISM", false);
    for (decltype(perf_results.size()) i = 0; i != perf_results.size(); ++i) {
      const auto &result = perf_results[i];
      bool algo_exclusion = static_cast<int32_t>(result.algo) == algo_exclude;
      bool algo_is_tensor_core = false;
      algo_is_tensor_core = result.mathType == CUDNN_TENSOR_OP_MATH;
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (!enforce_determinism || result.determinism == cudnnDeterminism_t::CUDNN_DETERMINISTIC) &&
          (param_.cudnn_tune.value() == conv::kLimited || result.memory <= workspace_byte) &&
          !algo_exclusion) {
        algo->Set(result.algo, algo_is_tensor_core);
        return;
      }
    }
    auto mode = param_.cudnn_tune.value() == conv::kOff ? " get " : " find ";
    LOG(FATAL) << "Failed to" << mode << "any " << kernel_name << " convolution algorithm. "
               << " with workspace size of " << workspace_byte << " bytes,"
               << " please consider reducing batch/model size or increasing the workspace size";
  }


  void GetTempSize(const RunContext& rctx) {
    mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               back_conv_desc_,
               in_desc_,
               back_algo_.AlgoNumber(),
               &back_workspace_byte_dgrad_));
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               back_conv_desc_w_,
               filter_desc_,
               back_algo_w_.AlgoNumber(),
               &back_workspace_byte_wgrad_));
    // cudaMalloc returns addresses that are aligned for large accesses (e.g. to 512 bytes).
    // Since we only make one allocation and divide it into two parts when we parallelize
    // the dgrad and wgrad kernels, we round the sizes up to this alignment size so the
    // dptrs respect this alignment, even if the separate areas are stacked.
    const size_t dptr_alignment = 512;
    back_workspace_byte_dgrad_ = RoundToMultiple(back_workspace_byte_dgrad_, dptr_alignment);
    back_workspace_byte_wgrad_ = RoundToMultiple(back_workspace_byte_wgrad_, dptr_alignment);

    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               forward_conv_desc_,
               out_desc_,
               forward_algo_.AlgoNumber(),
               &forward_workspace_byte_));
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
    CastTShapeToIntPtr(param_.pad, &param_pad_);
  }

  // Round a value 'x' up to the next multiple of 'multiple'
  size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx, size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words =
      std::max<size_t>(1, RoundToMultiple(size_bytes, sizeof(DType)) / sizeof(DType));
    return ctx.requested[conv::kTempSpace].get_space_typed<gpu, 1, DType>(
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
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return c;
  }

  // Make a number of allocations and directly free them, ensuring room for an equivalent set of
  // cudaMalloc() calls by (say) cudnnFind().  `elements` spec the alloc size in DTypes, not bytes.
  void ReserveElements(const std::vector<size_t> &elements) {
    std::vector<Storage::Handle> handles;
    for (size_t alloc_element : elements)
        handles.push_back(Storage::Get()->Alloc(alloc_element * sizeof(DType), Context::GPU()));
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
  std::vector<int> param_pad_;

  // Temp workspace size in bytes needed for Forward() operation.
  size_t forward_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() dgrad (data gradient) operation.
  size_t back_workspace_byte_dgrad_;
  // Temp workspace size in bytes needed for Backward() wgrad (weight gradient) operation.
  size_t back_workspace_byte_wgrad_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  // Convolution descriptor for forward inference operation
  cudnnConvolutionDescriptor_t forward_conv_desc_;
  // Convolution descriptor for back-prop operations to the data
  cudnnConvolutionDescriptor_t back_conv_desc_;
  // Convolution descriptor for back-prop operations to the weights
  cudnnConvolutionDescriptor_t back_conv_desc_w_;
  // Should dgrad and wgrad be launched into separate streams
  bool parallelize_backward_kernels_;
  // Algorithm for the forward inference operation
  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> forward_algo_;
  // Algorithm for the back-prop operation to the data
  CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> back_algo_;
  // Algorithm for the back-prop operation to the weights
  CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> back_algo_w_;
  cudnnTensorFormat_t format_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;
  // Is req[kWeight] == conv::kAddTo ?
  bool add_to_weight_;
  // Is there a dgrad algo that should be avoided (-1 == none)?
  int32_t exclude_dgrad_algo_ = -1;
  ConvolutionParam param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_CONVOLUTION_INL_H_
