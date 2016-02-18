/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm.cu
 * \brief
 * \author Junyuan Xie
*/

#include "./cudnn_batch_norm-inl.h"

namespace mxnet {
namespace op {
#if CUDNN_MAJOR >= 4
class CuDNNBatchNormOp : public Operator {
 public:
  explicit CuDNNBatchNormOp(CuDNNBatchNormParam param) {
    this->param_ = param;
    init_cudnn_ = false;
    dtype_ = CUDNN_DATA_FLOAT;
  }

  ~CuDNNBatchNormOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(io_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(mean_desc_), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3);
      CHECK_EQ(req.size(), 3);
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
    }
    CHECK_EQ(req[cudnnbatchnorm::kOut], kWriteTo);
    CHECK_GE(in_data[cudnnbatchnorm::kData].ndim(), 2);
    CHECK_LE(in_data[cudnnbatchnorm::kData].ndim(), 4);

    if (!init_cudnn_) {
      for (int i = 0; i < 4; ++i) {
        if (i < in_data[cudnnbatchnorm::kData].ndim()) {
          shape_[i] = in_data[cudnnbatchnorm::kData].shape_[i];
        } else {
          shape_[i] = 1;
        }
      }
      CHECK_EQ(cudnnCreateTensorDescriptor(&io_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnCreateTensorDescriptor(&mean_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(io_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          shape_[0],
                                          shape_[1],
                                          shape_[2],
                                          shape_[3]), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(mean_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          1,
                                          shape_[1],
                                          1,
                                          1), CUDNN_STATUS_SUCCESS);
      init_cudnn_  = true;
    }

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> x = in_data[cudnnbatchnorm::kData].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 1> gamma = in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> beta = in_data[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 4> y = out_data[cudnnbatchnorm::kOut].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 1> moving_mean = aux_states[cudnnbatchnorm::kMovingMean].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> moving_inv_var = aux_states[cudnnbatchnorm::kMovingInvVar].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    float a = 1.0f, b = 0.0f;
    if (ctx.is_train) {
      Tensor<gpu, 1> save_mean = out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
      Tensor<gpu, 1> save_inv_var = out_data[cudnnbatchnorm::kInvVar].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
      CHECK_EQ(cudnnBatchNormalizationForwardTraining(s->dnn_handle_,
                                                      CUDNN_BATCHNORM_SPATIAL,
                                                      &a,
                                                      &b,
                                                      io_desc_,
                                                      x.dptr_,
                                                      io_desc_,
                                                      y.dptr_,
                                                      mean_desc_,
                                                      gamma.dptr_,
                                                      beta.dptr_,
                                                      param_.momentum,
                                                      moving_mean.dptr_,
                                                      moving_inv_var.dptr_,
                                                      param_.eps,
                                                      save_mean.dptr_,
                                                      save_inv_var.dptr_), CUDNN_STATUS_SUCCESS);
    } else {
      CHECK_EQ(cudnnBatchNormalizationForwardInference(s->dnn_handle_,
                                                       CUDNN_BATCHNORM_SPATIAL,
                                                       &a,
                                                       &b,
                                                       io_desc_,
                                                       x.dptr_,
                                                       io_desc_,
                                                       y.dptr_,
                                                       mean_desc_,
                                                       gamma.dptr_,
                                                       beta.dptr_,
                                                       moving_mean.dptr_,
                                                       moving_inv_var.dptr_,
                                                       param_.eps), CUDNN_STATUS_SUCCESS);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_EQ(in_grad.size(), 3);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> x = in_data[cudnnbatchnorm::kData].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 4> dx = in_grad[cudnnbatchnorm::kData].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 4> dy = out_grad[cudnnbatchnorm::kOut].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 1> gamma = in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> dbeta = in_grad[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> dgamma = in_grad[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> save_mean = out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> save_inv_var = out_data[cudnnbatchnorm::kInvVar].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    float a = 1.0f, b = 0.0f;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
#if CUDNN_VERSION >= 4007
    CHECK_EQ(cudnnBatchNormalizationBackward(s->dnn_handle_,
                                             CUDNN_BATCHNORM_SPATIAL,
                                             &a,
                                             &b,
                                             &a,
                                             &b,
                                             io_desc_,
                                             x.dptr_,
                                             io_desc_,
                                             dy.dptr_,
                                             io_desc_,
                                             dx.dptr_,
                                             mean_desc_,
                                             gamma.dptr_,
                                             dgamma.dptr_,
                                             dbeta.dptr_,
                                             param_.eps,
                                             save_mean.dptr_,
                                             save_inv_var.dptr_), CUDNN_STATUS_SUCCESS);
#else  // CUDNN_VERSION < 4007
    CHECK_EQ(cudnnBatchNormalizationBackward(s->dnn_handle_,
                                             CUDNN_BATCHNORM_SPATIAL,
                                             &a,
                                             &b,
                                             io_desc_,
                                             x.dptr_,
                                             io_desc_,
                                             dy.dptr_,
                                             io_desc_,
                                             dx.dptr_,
                                             mean_desc_,
                                             gamma.dptr_,
                                             dgamma.dptr_,
                                             dbeta.dptr_,
                                             param_.eps,
                                             save_mean.dptr_,
                                             save_inv_var.dptr_), CUDNN_STATUS_SUCCESS);
#endif
    if (param_.fix_gamma) dgamma = 0;
  }

 private:
  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t io_desc_, mean_desc_;
  mshadow::Shape<4> shape_;
  CuDNNBatchNormParam param_;
};

template<>
Operator *CreateOp<gpu>(CuDNNBatchNormParam param) {
  return new CuDNNBatchNormOp(param);
}
#endif  // CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet

