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
class CuDNNLocalResponseNormOp : public Operator {
 public:
  explicit CuDNNLocalResponseNormOp(LRNParam param) {
    param_ = param;
    init_cudnn_ = false;
    // TODO(xxx): fp16
    dtype_ = CUDNN_DATA_FLOAT;
  }

  ~CuDNNLocalResponseNormOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyLRNDescriptor(lrn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS);
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
    float alpha = 1.0f;
    float beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> data = in_data[lrn_enum::kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> out = out_data[lrn_enum::kOut].get<gpu, 4, real_t>(s);
    if (!init_cudnn_) {
      this->Init(s, in_data, out_data);
    }
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CHECK_EQ(cudnnLRNCrossChannelForward(s->dnn_handle_,
                                         lrn_desc_,
                                         CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                         &alpha,
                                         shape_desc_,
                                         data.dptr_,
                                         &beta,
                                         shape_desc_,
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
    float alpha = 1.0f;
    float beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> grad = out_grad[lrn_enum::kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> data = in_data[lrn_enum::kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> output_data = out_data[lrn_enum::kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> input_grad = in_grad[lrn_enum::kData].get<gpu, 4, real_t>(s);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CHECK_EQ(cudnnLRNCrossChannelBackward(s->dnn_handle_,
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
                                          input_grad.dptr_), CUDNN_STATUS_SUCCESS);
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      Tensor<gpu, 4> data = in_data[lrn_enum::kData].get<gpu, 4, real_t>(s);
      Tensor<gpu, 4> out = out_data[lrn_enum::kOut].get<gpu, 4, real_t>(s);
      unsigned lrn_n = param_.nsize;
      double alpha = param_.alpha;
      double beta = param_.beta;
      double lrn_k = param_.knorm;
      CHECK_EQ(data.shape_, out.shape_);
      CHECK_EQ(cudnnCreateLRNDescriptor(&lrn_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetLRNDescriptor(lrn_desc_,
                                     lrn_n,
                                     alpha,
                                     beta,
                                     lrn_k), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&shape_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(shape_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]), CUDNN_STATUS_SUCCESS);
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
