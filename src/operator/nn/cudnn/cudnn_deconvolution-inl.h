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

#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include "../deconvolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../../../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

template<typename DType>
class CuDNNDeconvolutionOp : public Operator {
 public:
  explicit CuDNNDeconvolutionOp(DeconvolutionParam param,
                                int forward_compute_type,
                                int backward_compute_type,
                                const std::vector<TShape>& in_shape,
                                const std::vector<TShape>& out_shape,
                                const Context& ctx) {
    using namespace mshadow;
    this->param_ = param;
    InitBufferForParam();
    auto cudnn_forward_compute_type = convertToCuDNNDataType(forward_compute_type);
    auto cudnn_backward_compute_type = convertToCuDNNDataType(backward_compute_type);
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    init_cudnn_ = false;
    init_temp_size_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    // TensorCore algos only allowed on fp16-I/O deconvolutions if permitted by the global policy.
    cudnn_tensor_core_ = DataType<DType>::kFlag == kFloat16 && GetEnvAllowTensorCore();

#if CUDNN_MAJOR >= 5
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
#else
    CHECK(param_.layout.value() == kNCW ||
          param_.layout.value() == kNCHW ||
          param_.layout.value() == kNCDHW) << "Need CuDNN > 5.0 for layout support";
#endif
    // Double check to make sure this class supports the operation
    if (!Supports(param, forward_compute_type, backward_compute_type, ctx))
      LOG(FATAL) << "Need CuDNN >= 6.0 for dilated deconvolution.";

    InitDescriptors(ctx, in_shape, out_shape,
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
    SelectAlgo(ctx, in_shape, out_shape,
               cudnn_forward_compute_type, cudnn_backward_compute_type);
  }

  ~CuDNNDeconvolutionOp() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(forward_conv_desc_));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(back_conv_desc_));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(back_conv_desc_w_));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
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
      #if CUDNN_MAJOR <= 4
      CUDNN_CALL(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
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
                 out.dptr_ + out_offset_ * g));
      #elif CUDNN_MAJOR >= 5
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
      #endif
      if (!param_.no_bias) {
        beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[deconv::kBias].get<gpu, 1, DType>(s);
#if CUDNN_MAJOR >= 4
        CUDNN_CALL(cudnnAddTensor(s->dnn_handle_,
                                  &alpha,
                                  bias_desc_,
                                  bias.dptr_ + bias_offset_ * g,
                                  &beta,
                                  out_desc_,
                                  out_ptr + out_offset_ * g));
#endif
#if CUDNN_MAJOR == 3
        CUDNN_CALL(cudnnAddTensor(s->dnn_handle_,
                                  CUDNN_ADD_SAME_C,
                                  &alpha,
                                  bias_desc_,
                                  bias.dptr_ + bias_offset_ * g,
                                  &beta,
                                  out_desc_,
                                  out_ptr + out_offset_ * g));
#endif
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
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
      if (req[deconv::kWeight] != kNullOp) {
        #if CUDNN_MAJOR <= 4
        CUDNN_CALL(cudnnConvolutionBackwardFilter_v3(
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
          gwmat.dptr_ + weight_offset_ * g));
        #elif CUDNN_MAJOR >= 5
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
        #endif
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
                       const Context &ctx) {
    using namespace mshadow;

    // NDHWC not supported, NHWC not supported in true fp16
    auto layout_val = param.layout.value();
    auto true_fp16 = DataType<DType>::kFlag == kFloat16 &&
      (forward_compute_type == kFloat16 || backward_compute_type == kFloat16);
    if (layout_val == kNDHWC || layout_val == kNWC ||
        layout_val == kNHWC && true_fp16)
      return false;

    // Permits graceful fallback to pseudo-fp16 on heterogenous systems
    if (!SupportsFloat16Compute(ctx.dev_id) &&
        (forward_compute_type == kFloat16 || backward_compute_type == kFloat16)) {
      return false;
    }

    // The factor by which the effective filter size grows based on dilation.
    auto filterDilationFactor = param.dilate.Size();

    // The v6 kernels that backprop a dilated convolution don't handle fp16.
    // Since the deconvolution "forward" kernel is really a backprop-to-data
    // cuDNN kernel, the following logic is slightly different than that
    // used in CuDNNConvolution::Supports().

    // Dilation support across all architectures only available after v6.0.20.
    return filterDilationFactor == 1 ||
           filterDilationFactor > 1 && (CUDNN_VERSION > 6020) &&
           (backward_compute_type != kFloat16) &&
           (forward_compute_type != kFloat16);
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

  inline void InitDescriptors(const Context& ctx,
                              const std::vector<TShape> &in_shape,
                              const std::vector<TShape> &out_shape,
                              cudnnDataType_t cudnn_forward_compute_type,
                              cudnnDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&forward_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&back_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&back_conv_desc_w_));

    TShape dshape = in_shape[deconv::kData];
    TShape wshape = in_shape[deconv::kWeight];
    TShape oshape = out_shape[deconv::kOut];
    TShape dstride, ostride;
    wshape[0] /= param_.num_group;
#if CUDNN_MAJOR <= 5
      // As of cuDNN_v6, the unsuffixed version of cudnnSetConvolution2dDescriptor()
      // takes an additional 'computeType' parameter to set the precision of the
      // convolution calculation.  Supply this method signature for cuDNN versions < 6.
#define cudnnSetConvolution2dDescriptor(cdesc, p0, p1, s0, s1, d0, d1, m, ct) \
        cudnnSetConvolution2dDescriptor(cdesc, p0, p1, s0, s1, d0, d1, m)
#endif
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
      auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
      auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});

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
#if CUDNN_MAJOR < 5
      // As of cuDNN_v5, cudnnSetFilter4dDescriptor() takes a format parameter.
      // Supply this method signature for cuDNN versions < 5.
#define cudnnSetFilter4dDescriptor(fdesc, dt, f, w0, w1, w2, w3) \
        cudnnSetFilter4dDescriptor(fdesc, dt, w0, w1, w2, w3)
      CHECK_EQ(format_, CUDNN_TENSOR_NCHW) << "CuDNN V4 and earlier only supports NCHW layout";
#endif
      if (param_.kernel.ndim() == 2) {
        wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
        dstride = ConvertLayout(Strides<4>(dshape), param_.layout.value(), kNCHW);
        dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        ostride = ConvertLayout(Strides<4>(oshape), param_.layout.value(), kNCHW);
        oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
      } else {
        wshape = ConvertLayout(wshape.get<3>(), param_.layout.value(), kNCW);
        wshape = TShape({wshape[0], wshape[1], 1, wshape[2]});
        dstride = ConvertLayout(Strides<3>(dshape), param_.layout.value(), kNCW);
        dstride = TShape({dstride[0], dstride[1], dstride[1], dstride[2]});
        dshape = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
        dshape = TShape({dshape[0], dshape[1], 1, dshape[2]});
        ostride = ConvertLayout(Strides<3>(oshape), param_.layout.value(), kNCW);
        ostride = TShape({ostride[0], ostride[1], ostride[1], ostride[2]});
        oshape = ConvertLayout(oshape.get<3>(), param_.layout.value(), kNCW);
        oshape = TShape({oshape[0], oshape[1], 1, oshape[2]});
      }
      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                            dtype_,
                                            format_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      index_t o_pad[3];
      index_t o_adj[3];
      param_.InferPad(dshape, o_pad, o_adj);

      #if CUDNN_MAJOR >= 5
      CHECK_EQ(param_.layout.value(), kNCDHW) << "CuDNN only support 3D conv with NCDHW layout";
      std::vector<int> wshape_buffer(wshape.ndim());
      CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc_,
                                            dtype_,
                                            CUDNN_TENSOR_NCHW,
                                            static_cast<int>(wshape.ndim()),
                                            CastTShapeToIntPtr(wshape, &wshape_buffer)));
      #else
      LOG(FATAL) << "Only support CUDNN V5 for 3D convolution";
      #endif
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
#if CUDNN_MAJOR >= 7
    cudnnMathType_t math_type = cudnn_tensor_core_ ? CUDNN_TENSOR_OP_MATH
                                                  : CUDNN_DEFAULT_MATH;
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w_, math_type));
#endif
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
      TShape bias = in_shape[deconv::kBias];
      bias_offset_ = bias[0] / param_.num_group;
      std::vector<int> bias_shape = {1,
                                     static_cast<int>(bias[0] / param_.num_group),
                                     1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias_offset_), 1, 1, 1};
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
    init_cudnn_ = true;
  }

  void SelectAlgo(const Context& ctx,
                  const std::vector<TShape>& in_shape,
                  const std::vector<TShape>& out_shape,
                  cudnnDataType_t cudnn_forward_compute_type,
                  cudnnDataType_t cudnn_backward_compute_type) {
    if (!CuDNNDeconvAlgoReg::Get()->Find(param_, in_shape, out_shape, dtype_,
                                         cudnn_forward_compute_type,
                                         cudnn_backward_compute_type,
                                         SMArch(ctx.dev_id), &forward_algo_,
                                         &back_algo_, &back_algo_w_)) {
      // Not in algo registry, must determine via *Get*() or *Find*()
      Engine::VarHandle var = Engine::Get()->NewVariable();
      Engine::Get()->PushAsync([=](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        mshadow::Stream <gpu> *s = rctx.get_stream<gpu>();
        CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
        size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));
        #if CUDNN_MAJOR >= 7
          // Starting with cuDNNv7, the algo number returned by *Get*() is not the entire
          // story: the notion of whether the algo ran in Tensor Core mode is not known.
          // Since we want to report the Tensor Core mode in the verbose output, we switch
          // to using the new *Get*_v7() call.  Since the function signature of *Get*_v7() matches
          // that of *Find*(), we can unify the find-vs-get logic by using function pointers.

          // Forward Algorithm Find/Get() v7
          std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_results(MaxForwardAlgos(s->dnn_handle_));
          int actual_fwd_algos = 0;
          auto fwd_algo_discoverer =
            param_.cudnn_tune.value() == conv::kOff ? cudnnGetConvolutionForwardAlgorithm_v7
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
                                                     workspace_byte, &forward_algo_);

          // Backprop-to-Filter Algorithm Find/Get() v7
          auto max_bwd_filt_algos = MaxBackwardFilterAlgos(s->dnn_handle_);
          std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filt_results(max_bwd_filt_algos);
          int actual_bwd_filter_algos = 0;
          auto bwd_filter_algo_discoverer =
            param_.cudnn_tune.value() == conv::kOff ? cudnnGetConvolutionBackwardFilterAlgorithm_v7
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
                                                           workspace_byte, &back_algo_w_);

          // Backprop-to-Data Algorithm Find/Get() v7
          auto max_bwd_data_algos = MaxBackwardDataAlgos(s->dnn_handle_);
          std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_results(max_bwd_data_algos);
          int actual_bwd_data_algos = 0;
          auto bwd_data_algo_discoverer =
            param_.cudnn_tune.value() == conv::kOff ? cudnnGetConvolutionBackwardDataAlgorithm_v7
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
                                                         workspace_byte, &back_algo_);
        #else
        // CUDNN_MAJOR < 7
        const int kMaxAlgos = 10;
        int nalgo = kMaxAlgos;
        int i = 0;
        // Forward Algorithm Find/Get, v6 and earlier
        if (CUDNN_MAJOR == 6 && param_.layout.value() == mshadow::kNHWC) {
          // In cuDNNv6, for kNHWC, only CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM is
          // supported.  Hard-coded this since the algo find() or get() throws an FPE.
          forward_algo_.Set(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, false);
        } else if (!param_.cudnn_tune.value()) {
          cudnnConvolutionFwdAlgo_t fastest_fwd_algo;
          CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
                                                     out_desc_,
                                                     filter_desc_,
                                                     back_conv_desc_,  // fwd algo used in dgrad
                                                     in_desc_,
                                                     CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                     workspace_byte,
                                                     &fastest_fwd_algo));
          forward_algo_.Set(fastest_fwd_algo, false);
        } else {
          cudnnConvolutionFwdAlgoPerf_t fwd_algo[kMaxAlgos];
          CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(s->dnn_handle_,
                                                        out_desc_,
                                                        filter_desc_,
                                                        back_conv_desc_,  // fwd algo used in dgrad
                                                        in_desc_,
                                                        kMaxAlgos,
                                                        &nalgo,
                                                        fwd_algo));
          i = 0;
          while (i < nalgo
                 && (fwd_algo[i].status != CUDNN_STATUS_SUCCESS
                     || (param_.cudnn_tune.value() == deconv::kLimited
                         && fwd_algo[i].memory > workspace_byte)))
            ++i;
          if (i == nalgo) {
            LOG(FATAL) << "Failed to find a 'forward' convolution algorithm " <<
                       "(for use in deconvolution operator backprop-to-data).";
          } else {
            forward_algo_.Set(fwd_algo[i].algo, false);
          }
        }
        // Backprop-to-Filter Algorithm Find/Get, v6 and earlier
        if (!param_.cudnn_tune.value()) {
          cudnnConvolutionBwdFilterAlgo_t fastest_bwd_filt_algo;
          CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                                              out_desc_,
                                              in_desc_,
                                              back_conv_desc_,
                                              filter_desc_,
                                              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                              workspace_byte,
                                              &fastest_bwd_filt_algo));
          back_algo_w_.Set(fastest_bwd_filt_algo, false);
        } else {
          cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo[kMaxAlgos];
          CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                                                                 out_desc_,
                                                                 in_desc_,
                                                                 back_conv_desc_,
                                                                 filter_desc_,
                                                                 kMaxAlgos,
                                                                 &nalgo,
                                                                 bwd_filter_algo));
          i = 0;
          while (i < nalgo
                 && (bwd_filter_algo[i].status != CUDNN_STATUS_SUCCESS
                     || (param_.cudnn_tune.value() == deconv::kLimited
                         && bwd_filter_algo[i].memory > workspace_byte)))
            ++i;
          if (i == nalgo) {
            LOG(FATAL) << "Failed to find a backward filter convolution algorithm " <<
                       "(for use in deconvolution operator backprop-to-filter).";
          } else {
            back_algo_w_.Set(bwd_filter_algo[i].algo, false);
          }
        }
        // Backprop-to-Data Algorithm Get(), v6 and earlier
        if (!param_.cudnn_tune.value()) {
          cudnnConvolutionBwdDataAlgo_t fastest_bwd_data_algo;
          CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                                                filter_desc_,
                                                in_desc_,
                                                forward_conv_desc_,  // bwd algo used for inference
                                                out_desc_,
                                                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                                workspace_byte,
                                                &fastest_bwd_data_algo));
          back_algo_.Set(fastest_bwd_data_algo, false);
        } else {
          cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo[kMaxAlgos];
          CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                                                 filter_desc_,
                                                 in_desc_,
                                                 forward_conv_desc_,  // bwd algo used in inference
                                                 out_desc_,
                                                 kMaxAlgos,
                                                 &nalgo,
                                                 bwd_data_algo));
          i = 0;
          while (i < nalgo
                 && (bwd_data_algo[i].status != CUDNN_STATUS_SUCCESS
                     || (param_.cudnn_tune.value() == deconv::kLimited
                         && bwd_data_algo[i].memory > workspace_byte)))
            ++i;
          if (i == nalgo) {
            LOG(FATAL) << "Failed to find a backward data convolution algorithm." <<
                       "(for use in deconvolution operator forward inference).";
          } else {
            back_algo_.Set(bwd_data_algo[i].algo, false);
          }
        }
        #endif  // CUDNN_MAJOR < 7
        // An algo specification by the user may be cached here, but another
        // convolution will match only if identically specified.
        // We're caching results of *Get* as well as *Find*, but these records
        // will be held distinctly because param_.cudnn_tune is part of the key.
        CuDNNDeconvAlgoReg::Get()->Register(param_, in_shape, out_shape, dtype_,
                                            cudnn_forward_compute_type,
                                            cudnn_backward_compute_type,
                                            SMArch(ctx.dev_id), this->forward_algo_,
                                            this->back_algo_, this->back_algo_w_);
        on_complete();
      }, ctx, {}, {var});
      Engine::Get()->WaitForVar(var);
      Engine::Get()->DeleteVariable([](RunContext s) {}, ctx, var);
    }
    // If we're allowing Tensor Core variants of the algos to be considered in
    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    #if CUDNN_MAJOR >= 7
      // The next two code lines will look like they have typos, but they don't!
      // The forward_conv_desc_ is used during inference, which invokes the back_algo_.
      // Thus, the mathType of the back_algo_ should be stored in the forward_conv_desc_.
      // Conversely, the back_conv_desc_ is used during training backprop, which invokes
      // the forward_algo_.  Thus, the mathType of the forward_algo_ should be stored
      // in the back_conv_desc_.
      CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, back_algo_.MathType()));
      CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_, forward_algo_.MathType()));
      CUDNN_CALL(cudnnSetConvolutionMathType(back_conv_desc_w_, back_algo_w_.MathType()));
    #endif
  }

  // Look over the results from *Find*() or *Get*() and pick the fastest algo given possible
  // workspace constraints and a possible user algo preference.
  template <typename PerfType, typename AlgoType>
  void AlgoFinalSelect(const std::vector<PerfType> &perf_results, std::string kernel_name,
                       size_t workspace_byte, CuDNNAlgo<AlgoType> *algo) {
    // Determine the fastest acceptable algo regardless of mathType.
    for (decltype(perf_results.size()) i = 0; i != perf_results.size(); ++i) {
      const auto &result = perf_results[i];
      bool algo_is_tensor_core = false;
      #if CUDNN_MAJOR >= 7
        algo_is_tensor_core = result.mathType == CUDNN_TENSOR_OP_MATH;
      #endif
      if (result.status == CUDNN_STATUS_SUCCESS &&
          (param_.cudnn_tune.value() != conv::kLimited || result.memory <= workspace_byte)) {
        algo->Set(result.algo, algo_is_tensor_core);
        return;
      }
    }
    auto mode = param_.cudnn_tune.value() == conv::kOff ? " get " : " find ";
    LOG(FATAL) << "Failed to" << mode << "any " << kernel_name << " deconvolution algorithm.";
  }

  void GetTempSize(const OpContext& ctx) {
    if (init_temp_size_) return;
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
    init_temp_size_ = true;
  }

  int *CastTShapeToIntPtr(const TShape& s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = NULL;
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

  // Converts a TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const TShape &s) {
    uint32_t ndim = s.ndim();
    TShape strides(ndim);
    for (uint32_t i = 0; i != ndim; ++i)
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

  std::vector<int> param_stride_;
  std::vector<int> param_dilate_;

  bool init_cudnn_;
  bool init_temp_size_;
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
  DeconvolutionParam param_;
};
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_DECONVOLUTION_INL_H_
