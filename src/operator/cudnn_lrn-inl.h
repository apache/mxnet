/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_lrn-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_CUDNN_LRN_INL_H_
#define MXNET_OPERATOR_CUDNN_LRN_INL_H_
#include <vector>
#include "./lrn-inl.h"

namespace mxnet {
namespace op {
template<typename DType>
class CuDNNLocalResponseNormOp : public Operator {
 public:
  explicit CuDNNLocalResponseNormOp(LRNParam param) {
    param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
  }

  ~CuDNNLocalResponseNormOp() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(shape_desc_));
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
    CHECK_EQ(out_data.size(), 2U);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> data = in_data[lrn_enum::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> out = out_data[lrn_enum::kOut].get<gpu, 4, DType>(s);
    if (!init_cudnn_) {
      this->Init(s, in_data, out_data);
    }
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(cudnnLRNCrossChannelForward(s->dnn_handle_,
                                           lrn_desc_,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alpha,
                                           shape_desc_,
                                           data.dptr_,
                                           &beta,
                                           shape_desc_,
                                           out.dptr_));
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
    CHECK_EQ(out_data.size(), 2U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> grad = out_grad[lrn_enum::kOut].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> data = in_data[lrn_enum::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> output_data = out_data[lrn_enum::kOut].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> input_grad = in_grad[lrn_enum::kData].get<gpu, 4, DType>(s);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(cudnnLRNCrossChannelBackward(s->dnn_handle_,
                                            lrn_desc_,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            shape_desc_,
                                            output_data.dptr_,
                                            shape_desc_,
                                            grad.dptr_,
                                            shape_desc_,
                                            data.dptr_,
                                            &beta,
                                            shape_desc_,
                                            input_grad.dptr_));
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      Tensor<gpu, 4, DType> data = in_data[lrn_enum::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[lrn_enum::kOut].get<gpu, 4, DType>(s);
      unsigned lrn_n = param_.nsize;
      double alpha = param_.alpha;
      double beta = param_.beta;
      double lrn_k = param_.knorm;
      CHECK_EQ(data.shape_, out.shape_);
      CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc_));
      CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc_,
                                       lrn_n,
                                       alpha,
                                       beta,
                                       lrn_k));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc_));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(shape_desc_,
                                            CUDNN_TENSOR_NCHW,
                                            dtype_,
                                            data.shape_[0],
                                            data.shape_[1],
                                            data.shape_[2],
                                            data.shape_[3]));
    }
  }
  bool init_cudnn_;
  LRNParam param_;
  cudnnDataType_t dtype_;
  cudnnLRNDescriptor_t lrn_desc_;
  cudnnTensorDescriptor_t shape_desc_;
};  // class CuDNNLocalResponseNormOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUDNN_LRN_INL_H_
