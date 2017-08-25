/*!
 * Copyright (c) 2017 by Contributors
 * \file max_pool.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/
#include "./max_pool-inl.h"
#include <vector>
#include <algorithm>


namespace mxnet {
namespace op {

template<typename DType>
class CuDNNMaxPoolOp : public Operator {
 public:
  explicit CuDNNMaxPoolOp(MaxPoolParam param) {
    param_ = param;
    if (param_.layout == mshadow::kNCHW) {
      N = 0, H = 2, W = 3, C = 1;
      format_ = CUDNN_TENSOR_NCHW;
    } else if (param_.layout == mshadow::kNHWC) {
      N = 0, H = 1, W = 2, C = 3;
      format_ = CUDNN_TENSOR_NHWC;
    }
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    mode_ = CUDNN_POOLING_MAX;
  }

  ~CuDNNMaxPoolOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyPoolingDescriptor(max_pool_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;

    // 2d pool
    Tensor<gpu, 4, DType> data = in_data[pool_enum::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> out = out_data[pool_enum::kOut].get<gpu, 4, DType>(s);
    if (!init_cudnn_) {
      this->Init(s, in_data, out_data);
    }
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(cudnnPoolingForward(s->dnn_handle_,
                                 max_pool_desc_,
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
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;

    // 2d pool
    Tensor<gpu, 4, DType> m_out_grad = out_grad[pool_enum::kOut].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> m_in_data = in_data[pool_enum::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> m_out_data = out_data[pool_enum::kOut].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> m_in_grad = in_grad[pool_enum::kData].get<gpu, 4, DType>(s);
    CHECK_EQ(cudnnPoolingBackward(s->dnn_handle_,
                                  max_pool_desc_,
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
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      // 2d conv
      Tensor<gpu, 4, DType> data = in_data[pool_enum::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[pool_enum::kOut].get<gpu, 4, DType>(s);
      CHECK_EQ(cudnnCreatePoolingDescriptor(&max_pool_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(in_desc_,
                                          format_,
                                          dtype_,
                                          data.shape_[N],
                                          data.shape_[C],
                                          data.shape_[H],
                                          data.shape_[W]), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(out_desc_,
                                          format_,
                                          dtype_,
                                          out.shape_[N],
                                          out.shape_[C],
                                          out.shape_[H],
                                          out.shape_[W]), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetPooling2dDescriptor(max_pool_desc_,
                                           mode_,
                                           nan_prop_,
                                           param_.kernel[0],
                                           param_.kernel[1],
                                           param_.pad[0],
                                           param_.pad[1],
                                           param_.stride[0],
                                           param_.stride[1]),
                                           CUDNN_STATUS_SUCCESS);
    }
  }
  uint32_t N, H, W, C;
  bool init_cudnn_;
  cudnnHandle_t handle_;
  cudnnDataType_t dtype_;
  cudnnTensorFormat_t format_;
  cudnnPoolingMode_t mode_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingDescriptor_t max_pool_desc_;
  cudnnNanPropagation_t nan_prop_;
  MaxPoolParam param_;
};  // class CuDNNMaxPoolOp

template<>
Operator *CreateOp<gpu>(MaxPoolParam param, int dtype) {
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    return new CuDNNMaxPoolOp<DType>(param);
  });
}

}  // namespace op
}  // namespace mxnet

