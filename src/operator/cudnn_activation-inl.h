/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_activation-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_CUDNN_ACTIVATION_INL_H_
#define MXNET_OPERATOR_CUDNN_ACTIVATION_INL_H_
#include <algorithm>
#include <vector>
#include "./activation-inl.h"

namespace mxnet {
namespace op {
class CuDNNActivationOp : public Operator {
 public:
  explicit CuDNNActivationOp(ActivationParam param) {
    param_ = param;
    init_cudnn_ = false;
    dtype_ = CUDNN_DATA_FLOAT;
    switch (param_.act_type) {
      case kReLU:
        mode_ = CUDNN_ACTIVATION_RELU;
        break;
      case kSigmoid:
        mode_ = CUDNN_ACTIVATION_SIGMOID;
        break;
      case kTanh:
        mode_ = CUDNN_ACTIVATION_TANH;
        break;
      default:
        LOG(FATAL) << "Not implmented";
        break;
    }
  }

  ~CuDNNActivationOp() {
    CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS);
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
    Tensor<gpu, 4> data = in_data[kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> out = out_data[kOut].get<gpu, 4, real_t>(s);
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    if (!init_cudnn_) {
      this->Init(s, in_data, out_data);
    }
    CHECK_EQ(cudnnActivationForward(s->dnn_handle_,
                                    mode_,
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
    Tensor<gpu, 4> grad = out_grad[kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> data = in_data[kData].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> output_data = out_data[kOut].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> input_grad = in_grad[kData].get<gpu, 4, real_t>(s);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CHECK_EQ(cudnnActivationBackward(s->dnn_handle_,
                                     mode_,
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
      Tensor<gpu, 4> data = in_data[kData].get<gpu, 4, real_t>(s);
      Tensor<gpu, 4> out = out_data[kOut].get<gpu, 4, real_t>(s);
      CHECK_EQ(data.shape_, out.shape_);
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
  cudnnDataType_t dtype_;
  cudnnActivationMode_t mode_;
  cudnnTensorDescriptor_t shape_desc_;
  ActivationParam param_;
};  // class CuDNNActivationOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUDNN_ACTIVATION_INL_H_
