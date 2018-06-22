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
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_
#include <algorithm>
#include <vector>
#include "../pooling-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class CuDNNPoolingOp {
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
        mode_ = CUDNN_POOLING_MAX;
        break;
      case pool_enum::kAvgPooling:
        if (param_.count_include_pad.has_value() && !param_.count_include_pad.value()) {
          mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        } else {
          mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        }
        break;
      default:
        LOG(FATAL) << "Not implmented";
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
    this->Init(s, in_data, out_data);
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
      LOG(FATAL) << "Only support 2D or 3D pooling";
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
    this->Init(s, in_data, out_data);
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
      LOG(FATAL) << "Only support 2D or 3D pooling";
    }
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s, const TBlob &in_data,
      const TBlob &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR >= 5
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    #endif
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      Tensor<gpu, 4, DType> data = in_data.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data.get<gpu, 4, DType>(s);
      mshadow::Shape<4> dshape = data.shape_;
      CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                            CUDNN_TENSOR_NCHW,
                                            dtype_,
                                            data.shape_[0],
                                            data.shape_[1],
                                            data.shape_[2],
                                            data.shape_[3]));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                            CUDNN_TENSOR_NCHW,
                                            dtype_,
                                            out.shape_[0],
                                            out.shape_[1],
                                            out.shape_[2],
                                            out.shape_[3]));
      #if CUDNN_MAJOR >= 5
      CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_desc_,
                                             mode_,
                                             nan_prop_,
                                             param_.global_pool ? dshape[2] : param_.kernel[0],
                                             param_.global_pool ? dshape[3] : param_.kernel[1],
                                             param_.global_pool ? 0 : param_.pad[0],
                                             param_.global_pool ? 0 : param_.pad[1],
                                             param_.global_pool ? 1 : param_.stride[0],
                                             param_.global_pool ? 1 :param_.stride[1]));
      #else
      CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_desc_,
                                             mode_,
                                             param_.global_pool ? dshape[2] : param_.kernel[0],
                                             param_.global_pool ? dshape[3] : param_.kernel[1],
                                             param_.global_pool ? 0 : param_.pad[0],
                                             param_.global_ppol ? 0 : param_.pad[1],
                                             param_.global_pool ? 1 : param_.stride[0],
                                             param_.global_pool ? 1 : param_.stride[1]));
      #endif
    } else {
      Tensor<gpu, 5, DType> data = in_data.get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data.get<gpu, 5, DType>(s);
      std::vector<int> ishape = {static_cast<int>(data.shape_[0]),
                                 static_cast<int>(data.shape_[1]),
                                 static_cast<int>(data.shape_[2]),
                                 static_cast<int>(data.shape_[3]),
                                 static_cast<int>(data.shape_[4])};

      std::vector<int> istride = {static_cast<int>(ishape[1] * ishape[2] * ishape[3] * ishape[4]),
                                  static_cast<int>(ishape[2] * ishape[3] * ishape[4]),
                                  static_cast<int>(ishape[3] * ishape[4]),
                                  static_cast<int>(ishape[4]), 1};

      std::vector<int> oshape = {static_cast<int>(out.shape_[0]),
                                 static_cast<int>(out.shape_[1]),
                                 static_cast<int>(out.shape_[2]),
                                 static_cast<int>(out.shape_[3]),
                                 static_cast<int>(out.shape_[4])};

      std::vector<int> ostride = {static_cast<int>(oshape[1] * oshape[2] * oshape[3] * oshape[4]),
                                  static_cast<int>(oshape[2] * oshape[3] * oshape[4]),
                                  static_cast<int>(oshape[3] * oshape[4]),
                                  static_cast<int>(oshape[4]), 1};

      std::vector<int> kernel_vec = {param_.global_pool ? ishape[2] :
                                                          static_cast<int>(param_.kernel[0]),
                                     param_.global_pool ? ishape[3] :
                                                          static_cast<int>(param_.kernel[1]),
                                     param_.global_pool ? ishape[4] :
                                                          static_cast<int>(param_.kernel[2])};

      std::vector<int> pad_vec = {param_.global_pool ? 0 : static_cast<int>(param_.pad[0]),
                                  param_.global_pool ? 0 : static_cast<int>(param_.pad[1]),
                                  param_.global_pool ? 0 : static_cast<int>(param_.pad[2])};

      std::vector<int> stride_vec = {param_.global_pool ? 1 : static_cast<int>(param_.stride[0]),
                                     param_.global_pool ? 1 : static_cast<int>(param_.stride[1]),
                                     param_.global_pool ? 1 : static_cast<int>(param_.stride[2])};

      CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                            dtype_,
                                            static_cast<int>(ishape.size()),
                                            &ishape[0],
                                            &istride[0]));
      CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                            dtype_,
                                            static_cast<int>(oshape.size()),
                                            &oshape[0],
                                            &ostride[0]));
      #if CUDNN_MAJOR >= 5
      CUDNN_CALL(cudnnSetPoolingNdDescriptor(pooling_desc_,
                                             mode_,
                                             nan_prop_,
                                             static_cast<int>(kernel_vec.size()),
                                             &(kernel_vec[0]),
                                             &(pad_vec[0]),
                                             &(stride_vec[0])));
      #else
      LOG(FATAL) << "3D pooling only support CUDNN v5 and above";
      #endif
    }
  }

  cudnnDataType_t dtype_;
  cudnnHandle_t handle_;
  cudnnPoolingMode_t mode_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  #if CUDNN_MAJOR >= 5
  cudnnNanPropagation_t nan_prop_;
  #endif
  PoolingParam param_;
};  // class CuDNNPoolingOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_POOLING_INL_H_

