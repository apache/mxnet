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

template<typename DType>
class CuDNNPoolingOp : public Operator {
 public:
  explicit CuDNNPoolingOp(PoolingParam p) {
    param_ = p;
    init_cudnn_ = false;
    // TODO(xxx): fp16
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
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
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    if (param_.kernel.ndim() == 2) {
      // 2d pool
      Tensor<gpu, 4, DType> data = in_data[pool_enum::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[pool_enum::kOut].get<gpu, 4, DType>(s);
      if (!init_cudnn_) {
        this->Init(s, in_data, out_data);
      }
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
    } else if (param_.kernel.ndim() == 3) {
      // 3d pool
      Tensor<gpu, 5, DType> data = in_data[pool_enum::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data[pool_enum::kOut].get<gpu, 5, DType>(s);
      if (!init_cudnn_) {
        this->Init(s, in_data, out_data);
      }
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
    } else {
      LOG(FATAL) << "Only support 2D or 3D pooling";
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    if (param_.kernel.ndim() == 2) {
      // 2d pool
      Tensor<gpu, 4, DType> m_out_grad = out_grad[pool_enum::kOut].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_in_data = in_data[pool_enum::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_out_data = out_data[pool_enum::kOut].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> m_in_grad = in_grad[pool_enum::kData].get<gpu, 4, DType>(s);
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
    } else if (param_.kernel.ndim() == 3) {
      // 3d pool
      Tensor<gpu, 5, DType> m_out_grad = out_grad[pool_enum::kOut].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_in_data = in_data[pool_enum::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_out_data = out_data[pool_enum::kOut].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> m_in_grad = in_grad[pool_enum::kData].get<gpu, 5, DType>(s);
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
    } else {
      LOG(FATAL) << "Only support 2D or 3D pooling";
    }
  }

 private:
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
      if (param_.kernel.ndim() == 2) {
        // 2d conv
        Tensor<gpu, 4, DType> data = in_data[pool_enum::kData].get<gpu, 4, DType>(s);
        Tensor<gpu, 4, DType> out = out_data[pool_enum::kOut].get<gpu, 4, DType>(s);
        mshadow::Shape<4> dshape = data.shape_;
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
      } else {
        Tensor<gpu, 5, DType> data = in_data[pool_enum::kData].get<gpu, 5, DType>(s);
        Tensor<gpu, 5, DType> out = out_data[pool_enum::kOut].get<gpu, 5, DType>(s);
        CHECK_EQ(cudnnCreatePoolingDescriptor(&pooling_desc_), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
        std::vector<int> ishape = {static_cast<int>(data.shape_[0]),
                                   static_cast<int>(data.shape_[1]),
                                   static_cast<int>(data.shape_[2]),
                                   static_cast<int>(data.shape_[3]),
                                   static_cast<int>(data.shape_[4])};

        std::vector<int> istride = {static_cast<int>(ishape[1] * ishape[2] * ishape[3] * ishape[4]),
                                    static_cast<int>(ishape[2] * ishape[3] * ishape[4]),
                                    static_cast<int>(ishape[3] * ishape[4]),
                                    static_cast<int>(ishape[4]),
                                    1};

        std::vector<int> oshape = {static_cast<int>(out.shape_[0]),
                                   static_cast<int>(out.shape_[1]),
                                   static_cast<int>(out.shape_[2]),
                                   static_cast<int>(out.shape_[3]),
                                   static_cast<int>(out.shape_[4])};

        std::vector<int> ostride = {static_cast<int>(oshape[1] * oshape[2] * oshape[3] * oshape[4]),
                                    static_cast<int>(oshape[2] * oshape[3] * oshape[4]),
                                    static_cast<int>(oshape[3] * oshape[4]),
                                    static_cast<int>(oshape[4]),
                                    1};

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

        CHECK_EQ(cudnnSetTensorNdDescriptor(in_desc_,
                                            dtype_,
                                            static_cast<int>(ishape.size()),
                                            &ishape[0],
                                            &istride[0]), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnSetTensorNdDescriptor(out_desc_,
                                            dtype_,
                                            static_cast<int>(oshape.size()),
                                            &oshape[0],
                                            &ostride[0]), CUDNN_STATUS_SUCCESS);
        #if CUDNN_MAJOR == 5
        CHECK_EQ(cudnnSetPoolingNdDescriptor(pooling_desc_,
                                             mode_,
                                             nan_prop_,
                                             static_cast<int>(kernel_vec.size()),
                                             &(kernel_vec[0]),
                                             &(pad_vec[0]),
                                             &(stride_vec[0])), CUDNN_STATUS_SUCCESS);
        #else
        LOG(FATAL) << "3D pooling only support CUDNN v5 and abouve";
        #endif
      }
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

