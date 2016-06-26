/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_CUDNN_POOLING_INL_H_
#define MXNET_OPERATOR_CUDNN_POOLING_INL_H_
#include <algorithm>
#include <vector>
#include "./pooling-inl.h"

namespace mxnet {
namespace op {

class CuDNNPoolingOp : public Operator {
 public:
  explicit CuDNNPoolingOp(PoolingParam p) {
    param_ = p;
    init_cudnn_ = false;
    // TODO(xxx): fp16
    dtype_ = CUDNN_DATA_FLOAT;
    switch (param_.pool_type) {
      case pool_enum::kMaxPooling:
        mode_ = CUDNN_POOLING_MAX;
        break;
      case pool_enum::kAvgPooling:
        mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  ~CuDNNPoolingOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyPoolingDescriptor(pooling_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> data = in_data[pool_enum::kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> out = out_data[pool_enum::kOut].get<gpu, 4, real_t>(s);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    if (!init_cudnn_) {
      this->Init(s, in_data, out_data);
    }
    if (param_.global_pool) {
      this->InitGlobalPool(data.shape_);
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(cudnnPoolingForward(s->dnn_handle_,
                                 pooling_desc_,
                                 &alpha,
                                 in_desc_,
                                 data.dptr_,
                                 &beta,
                                 out_desc_,
                                 out.dptr_), CUDNN_STATUS_SUCCESS);
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> m_out_grad = out_grad[pool_enum::kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> m_in_data = in_data[pool_enum::kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> m_out_data = out_data[pool_enum::kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> m_in_grad = in_grad[pool_enum::kData].get<gpu, 4, real_t>(s);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_EQ(cudnnPoolingBackward(s->dnn_handle_,
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
                                  m_in_grad.dptr_), CUDNN_STATUS_SUCCESS);
  }

 private:
  inline void InitGlobalPool(const mshadow::Shape<4> &dshape) {
    #if CUDNN_MAJOR == 5
      CHECK_EQ(cudnnSetPooling2dDescriptor(pooling_desc_,
                                           mode_,
                                           nan_prop_,
                                           param_.global_pool ? dshape[2] : param_.kernel[0],
                                           param_.global_pool ? dshape[3] : param_.kernel[1],
                                           param_.pad[0],
                                           param_.pad[1],
                                           param_.global_pool ? 1 : param_.stride[0],
                                           param_.global_pool ? 1 :param_.stride[1]),
               CUDNN_STATUS_SUCCESS);
      #else
      CHECK_EQ(cudnnSetPooling2dDescriptor(pooling_desc_,
                                           mode_,
                                           param_.global_pool ? dshape[2] : param_.kernel[0],
                                           param_.global_pool ? dshape[3] : param_.kernel[1],
                                           param_.pad[0],
                                           param_.pad[1],
                                           param_.global_pool ? 1 : param_.stride[0],
                                           param_.global_pool ? 1 : param_.stride[1]),
               CUDNN_STATUS_SUCCESS);
      #endif
  }

  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR == 5
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    #endif
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      Tensor<gpu, 4> data = in_data[pool_enum::kData].get<gpu, 4, real_t>(s);
      Tensor<gpu, 4> out = out_data[pool_enum::kOut].get<gpu, 4, real_t>(s);
      CHECK_EQ(cudnnCreatePoolingDescriptor(&pooling_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(in_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(out_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          out.shape_[0],
                                          out.shape_[1],
                                          out.shape_[2],
                                          out.shape_[3]), CUDNN_STATUS_SUCCESS);
      #if CUDNN_MAJOR == 5
      CHECK_EQ(cudnnSetPooling2dDescriptor(pooling_desc_,
                                           mode_,
                                           nan_prop_,
                                           param_.kernel[0],
                                           param_.kernel[1],
                                           param_.pad[0],
                                           param_.pad[1],
                                           param_.stride[0],
                                           param_.stride[1]), CUDNN_STATUS_SUCCESS);
      #else
      CHECK_EQ(cudnnSetPooling2dDescriptor(pooling_desc_,
                                           mode_,
                                           param_.kernel[0],
                                           param_.kernel[1],
                                           param_.pad[0],
                                           param_.pad[1],
                                           param_.stride[0],
                                           param_.stride[1]), CUDNN_STATUS_SUCCESS);
      #endif
    }
  }
  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnHandle_t handle_;
  cudnnPoolingMode_t mode_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  #if CUDNN_MAJOR == 5
  cudnnNanPropagation_t nan_prop_;
  #endif
  PoolingParam param_;
};  // class CuDNNPoolingOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_POOLING_INL_H_

