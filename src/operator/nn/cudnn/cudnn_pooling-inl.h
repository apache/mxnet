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
 * \file cudnn_pooling-inl.h
 * \brief
 * \author Bing Xu, Dick Carter
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_
#include <algorithm>
#include <array>
#include "../pooling-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class CuDNNPoolingOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);

 public:
  CuDNNPoolingOp() {
    // TODO(xxx): fp16
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
  }

  void Init(const PoolingParam &p) {
    param_ = p;
    switch (param_.pool_type) {
      case pool_enum::kMaxPooling:
        mode_ = dmlc::GetEnv("MXNET_ENFORCE_DETERMINISM", false) ?
          CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_MAX;
        break;
      case pool_enum::kAvgPooling:
        if (param_.count_include_pad.has_value() && !param_.count_include_pad.value()) {
          mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        } else {
          mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        }
        break;
      default:
        LOG(FATAL) << "Pooling type not implemented by cuDNN.";
    }
  }

  ~CuDNNPoolingOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }

  void Forward(const OpContext &ctx, const TBlob &in_data,
      const OpReqType &req, const TBlob &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    CHECK(this->Init(s, in_data, out_data)) << "cuDNN Pooling invoked with unsupported parameters.";
    if (param_.kernel.ndim() == 2) {
      // 2d pool
      Tensor<gpu, 4, DType> data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      CUDNN_CALL(cudnnPoolingForward(s->dnn_handle_,
                                     pooling_desc_,
                                     &alpha,
                                     in_desc_,
                                     data.dptr_,
                                     &beta,
                                     out_desc_,
                                     out.dptr_));
    } else if (param_.kernel.ndim() == 3) {
      // 3d pool
      Tensor<gpu, 5, DType> data = in_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      CUDNN_CALL(cudnnPoolingForward(s->dnn_handle_,
                                     pooling_desc_,
                                     &alpha,
                                     in_desc_,
                                     data.dptr_,
                                     &beta,
                                     out_desc_,
                                     out.dptr_));
    } else {
      LOG(FATAL) << "cuDNN only supports 2D or 3D pooling.";
    }
  }

  void Backward(const OpContext &ctx, const TBlob &out_grad,
      const TBlob &in_data, const TBlob &out_data,
      const OpReqType &req, const TBlob &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;

    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    CHECK(this->Init(s, in_data, out_data)) << "cuDNN Pooling invoked with unsupported parameters.";
    if (param_.kernel.ndim() == 2) {
      // 2d pool
      Tensor<gpu, 4, DType> m_out_grad = out_grad.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_in_data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_out_data = out_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_in_grad = in_grad.get<gpu, 4, DType>(s);
      CUDNN_CALL(cudnnPoolingBackward(s->dnn_handle_,
                                      pooling_desc_,
                                      &alpha,
                                      out_desc_,
                                      m_out_data.dptr_,
                                      out_desc_,
                                      m_out_grad.dptr_,
                                      in_desc_,
                                      m_in_data.dptr_,
                                      &beta,
                                      in_desc_,
                                      m_in_grad.dptr_));
    } else if (param_.kernel.ndim() == 3) {
      // 3d pool
      Tensor<gpu, 5, DType> m_out_grad = out_grad.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_in_data = in_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_out_data = out_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_in_grad = in_grad.get<gpu, 5, DType>(s);
      CUDNN_CALL(cudnnPoolingBackward(s->dnn_handle_,
                                      pooling_desc_,
                                      &alpha,
                                      out_desc_,
                                      m_out_data.dptr_,
                                      out_desc_,
                                      m_out_grad.dptr_,
                                      in_desc_,
                                      m_in_data.dptr_,
                                      &beta,
                                      in_desc_,
                                      m_in_grad.dptr_));
    } else {
      LOG(FATAL) << "cuDNN only supports 2D or 3D pooling.";
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the pooling operation
 * described by `param`: cuDNN v5 and earlier does not support 3D pooling for example.
 * CuDNN v7.1.4 backprop kernel doesn't support kernel sizes 9 and above.
 */
  static bool Supports(const PoolingParam &param, const TBlob& input) {
    using namespace mshadow;
    static bool sum_pooling_warning_issued = false;
    static bool lp_pooling_warning_issued = false;
    static bool unsupported_dim_warning_issued = false;
    int layout = param.GetLayout(input.ndim());

    switch (param.pool_type) {
      case pool_enum::kMaxPooling:
      case pool_enum::kAvgPooling:
        break;
      case pool_enum::kSumPooling:
        if (!sum_pooling_warning_issued) {
          sum_pooling_warning_issued = true;
          LOG(WARNING) << "Sum pooling is not supported by cudnn, MXNet sum pooling is applied.";
        }
        return false;
      case pool_enum::kLpPooling:
        if (!lp_pooling_warning_issued) {
          lp_pooling_warning_issued = true;
          LOG(WARNING) << "Lp pooling is not supported by cudnn, MXNet Lp pooling is applied.";
        }
        return false;
      default:
        return false;
    }

    if (param.kernel.ndim() == 2) {
      // 2d pooling
      if (!(layout == mshadow::kNCHW || layout == mshadow::kNHWC))
        return false;
#if CUDNN_VERSION == 7104
      // CuDNN v7.1.4 backprop kernel doesn't support kernel sizes 9 and above.
      // Perform shape calculations in a standard (NCHW) layout space
      mshadow::Shape<4> input_shape = input.shape_.get<4>();
      mshadow::Shape<4> dshape_nchw = (layout == mshadow::kNHWC) ?
                                      ConvertLayout(input_shape, mshadow::kNHWC, mshadow::kNCHW) :
                                      input_shape;
      int kernel_height = param.global_pool ? dshape_nchw[2] : param.kernel[0];
      int kernel_width = param.global_pool ? dshape_nchw[3] : param.kernel[1];
      if (kernel_height > 8 || kernel_width > 8)
        return false;
#endif
#if CUDNN_VERSION >= 7105 && CUDNN_VERSION < 7500
      // Avoid strided NHWC max pooling for some configs
      if (layout == mshadow::kNHWC &&
          param.pool_type == pool_enum::kMaxPooling && !param.global_pool) {
        if (param.stride[0] >= 3 ||
            param.stride[0] == 2 && param.kernel[0] % 2 == 0 && param.kernel[0] != 2)
          return false;
        if (param.stride[1] >= 3 ||
            param.stride[1] == 2 && param.kernel[1] % 2 == 0 && param.kernel[1] != 2)
          return false;
      }
#endif
    } else if (param.kernel.ndim() == 3) {
      // 3d pooling
      if (!(layout == mshadow::kNCDHW || layout == mshadow::kNDHWC))
        return false;
    } else {
      // Unsupported kernel dim
      LogUnsupportedDim(&unsupported_dim_warning_issued, param.kernel.ndim());
      return false;
    }

    return true;
  }

 private:
  // Return boolean saying whether pooling configuration is supported
  inline bool Init(mshadow::Stream<gpu> *s, const TBlob &in_data,
      const TBlob &out_data) {
    using namespace mshadow;
    bool is_supported = true;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    int layout = param_.GetLayout(in_data.ndim());
    if (param_.kernel.ndim() == 2) {
      // 2d pooling
      CHECK(layout == mshadow::kNCHW || layout == mshadow::kNHWC) << "Need 2D layout NCHW or NHWC.";
      cudnnTensorFormat_t cudnn_layout = (layout == mshadow::kNCHW) ? CUDNN_TENSOR_NCHW
                                                                    : CUDNN_TENSOR_NHWC;
      Tensor<gpu, 4, DType> data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data.get<gpu, 4, DType>(s);
      // Perform shape calculations in a standard (NCHW) layout space
      mshadow::Shape<4> dshape_nchw = (layout == mshadow::kNHWC) ?
                                      ConvertLayout(data.shape_, mshadow::kNHWC, mshadow::kNCHW) :
                                      data.shape_;
      mshadow::Shape<4> oshape_nchw = (layout == mshadow::kNHWC) ?
                                      ConvertLayout(out.shape_, mshadow::kNHWC, mshadow::kNCHW) :
                                      out.shape_;
      CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                            cudnn_layout,
                                            dtype_,
                                            dshape_nchw[0],
                                            dshape_nchw[1],
                                            dshape_nchw[2],
                                            dshape_nchw[3]));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                            cudnn_layout,
                                            dtype_,
                                            oshape_nchw[0],
                                            oshape_nchw[1],
                                            oshape_nchw[2],
                                            oshape_nchw[3]));
      int kernel_height = param_.global_pool ? dshape_nchw[2] : param_.kernel[0];
      int kernel_width = param_.global_pool ? dshape_nchw[3] : param_.kernel[1];
      // CuDNN v7.1.4 backprop kernel doesn't support kernel sizes 9 and above.
      // For reference see Fixed Issues section in
      // https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/rel_721.html#rel_721
      #if CUDNN_VERSION == 7104
      is_supported = kernel_height <= 8 && kernel_width <= 8;
      #endif
      CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_desc_,
                                             mode_,
                                             nan_prop_,
                                             kernel_height,
                                             kernel_width,
                                             param_.global_pool ? 0 : param_.pad[0],
                                             param_.global_pool ? 0 : param_.pad[1],
                                             param_.global_pool ? 1 : param_.stride[0],
                                             param_.global_pool ? 1 : param_.stride[1]));
    } else {
      CHECK(layout == mshadow::kNCDHW ||
            layout == mshadow::kNDHWC) << "Need 3D layout NCDHW or NDHWC.";
      Tensor<gpu, 5, DType> data = in_data.get<gpu, 5, DType>(s);
      mshadow::Shape<5> dshape = data.shape_;
      mshadow::Shape<5> dstride = mshadow::Shape5(dshape.ProdShape(1, 5),
                                           dshape.ProdShape(2, 5),
                                           dshape.ProdShape(3, 5),
                                           dshape.ProdShape(4, 5),
                                           dshape.ProdShape(5, 5));

      Tensor<gpu, 5, DType> out = out_data.get<gpu, 5, DType>(s);
      mshadow::Shape<5> oshape = out.shape_;
      mshadow::Shape<5> ostride = mshadow::Shape5(oshape.ProdShape(1, 5),
                                           oshape.ProdShape(2, 5),
                                           oshape.ProdShape(3, 5),
                                           oshape.ProdShape(4, 5),
                                           oshape.ProdShape(5, 5));
      // Convert to a standard (NCDHW) layout space to create args for cuDNN

      mshadow::Shape<5> dshape_ncdhw = (layout == mshadow::kNDHWC) ?
                                       ConvertLayout(dshape, mshadow::kNDHWC, mshadow::kNCDHW) :
                                       dshape;
      mshadow::Shape<5> dstride_ncdhw = (layout == mshadow::kNDHWC) ?
                                        ConvertLayout(dstride, mshadow::kNDHWC, mshadow::kNCDHW) :
                                        dstride;
      mshadow::Shape<5> oshape_ncdhw = (layout == mshadow::kNDHWC) ?
                                        ConvertLayout(oshape, mshadow::kNDHWC, mshadow::kNCDHW) :
                                        oshape;
      mshadow::Shape<5> ostride_ncdhw = (layout == mshadow::kNDHWC) ?
                                        ConvertLayout(ostride, mshadow::kNDHWC, mshadow::kNCDHW) :
                                        ostride;
      // Create int arrays for passing into cuDNN
      std::array<int, 5> dshape_ncdhw_int, dstride_ncdhw_int, oshape_ncdhw_int, ostride_ncdhw_int;
      for (int i = 0; i < 5; ++i) {
        dshape_ncdhw_int[i] = static_cast<int>(dshape_ncdhw[i]);
        dstride_ncdhw_int[i] = static_cast<int>(dstride_ncdhw[i]);
        oshape_ncdhw_int[i] = static_cast<int>(oshape_ncdhw[i]);
        ostride_ncdhw_int[i] = static_cast<int>(ostride_ncdhw[i]);
      }

      std::array<int, 3> kernel_vec = {param_.global_pool ? static_cast<int>(dshape_ncdhw[2]) :
                                                          static_cast<int>(param_.kernel[0]),
                                     param_.global_pool ? static_cast<int>(dshape_ncdhw[3]) :
                                                          static_cast<int>(param_.kernel[1]),
                                     param_.global_pool ? static_cast<int>(dshape_ncdhw[4]) :
                                                          static_cast<int>(param_.kernel[2])};

      std::array<int, 3> pad_vec = {param_.global_pool ? 0 : static_cast<int>(param_.pad[0]),
                                  param_.global_pool ? 0 : static_cast<int>(param_.pad[1]),
                                  param_.global_pool ? 0 : static_cast<int>(param_.pad[2])};

      std::array<int, 3> stride_vec = {param_.global_pool ? 1 : static_cast<int>(param_.stride[0]),
                                     param_.global_pool ? 1 : static_cast<int>(param_.stride[1]),
                                     param_.global_pool ? 1 : static_cast<int>(param_.stride[2])};

      CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                            dtype_,
                                            static_cast<int>(dshape_ncdhw_int.size()),
                                            &dshape_ncdhw_int[0],
                                            &dstride_ncdhw_int[0]));
      CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                            dtype_,
                                            static_cast<int>(oshape_ncdhw_int.size()),
                                            &oshape_ncdhw_int[0],
                                            &ostride_ncdhw_int[0]));
      CUDNN_CALL(cudnnSetPoolingNdDescriptor(pooling_desc_,
                                             mode_,
                                             nan_prop_,
                                             static_cast<int>(kernel_vec.size()),
                                             &(kernel_vec[0]),
                                             &(pad_vec[0]),
                                             &(stride_vec[0])));
    }
    return is_supported;
  }

  // Log once that the dimension of the pooling operation isn't supported
  static void LogUnsupportedDim(bool *msg_logged, int ndim) {
    if (!*msg_logged) {
      *msg_logged = true;
      LOG(WARNING) << ndim << "D pooling is not supported by cudnn, "
                   << "MXNet " << ndim << "D pooling is applied.";
    }
  }

  cudnnDataType_t dtype_;
  cudnnHandle_t handle_;
  cudnnPoolingMode_t mode_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnNanPropagation_t nan_prop_;
  PoolingParam param_;
};  // class CuDNNPoolingOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_

