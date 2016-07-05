/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_CUDNN_BATCH_NORM_INL_H_
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "./batch_norm-inl.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4
namespace cudnnbatchnorm {
enum CuDNNBatchNormOpInputs {kData, kGamma, kBeta};
enum CuDNNBatchNormOpOutputs {kOut, kMean, kInvVar};
enum CuDNNBatchNormOpAuxiliary {kMovingMean, kMovingInvVar};
}  // namespace cudnnbatchnorm

#if defined(__CUDACC__)
class CuDNNBatchNormOp : public Operator {
 public:
  explicit CuDNNBatchNormOp(BatchNormParam param) {
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
    Tensor<gpu, 1> gamma =
      in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    if (param_.fix_gamma) gamma = 1.0f;
    Tensor<gpu, 1> beta =
      in_data[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 4> y = out_data[cudnnbatchnorm::kOut].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 1> moving_mean =
      aux_states[cudnnbatchnorm::kMovingMean].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> moving_inv_var =
      aux_states[cudnnbatchnorm::kMovingInvVar]
      .get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    float a = 1.0f, b = 0.0f;
    if (ctx.is_train) {
      Tensor<gpu, 1> save_mean =
        out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
      Tensor<gpu, 1> save_inv_var =
        out_data[cudnnbatchnorm::kInvVar].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
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
                                                      1 - param_.momentum,
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
    CHECK(ctx.is_train && !param_.use_global_stats)
        << "use global statistics is not yet supported in CuDNNBatchNorm";

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> x = in_data[cudnnbatchnorm::kData].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 4> dx = in_grad[cudnnbatchnorm::kData].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 4> dy = out_grad[cudnnbatchnorm::kOut].get_with_shape<gpu, 4, real_t>(shape_, s);
    Tensor<gpu, 1> gamma =
      in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> dbeta =
      in_grad[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> dgamma =
      in_grad[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> save_mean =
      out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    Tensor<gpu, 1> save_inv_var =
      out_data[cudnnbatchnorm::kInvVar].get_with_shape<gpu, 1, real_t>(Shape1(shape_[1]), s);
    float a = 1.0f;
    float b = 0.0f;
    float b_add = 1.0f;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
#if CUDNN_VERSION >= 4007
    CHECK_EQ(cudnnBatchNormalizationBackward(s->dnn_handle_,
                                             CUDNN_BATCHNORM_SPATIAL,
                                             &a,
                                             &b,
                                             &a,
                                             req[cudnnbatchnorm::kGamma] == kWriteTo ? &b: &b_add,
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
    if (param_.fix_gamma) dgamma = 0.f;
  }

 private:
  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t io_desc_, mean_desc_;
  mshadow::Shape<4> shape_;
  BatchNormParam param_;
};
#endif  // defined(__CUDACC__)

template<typename xpu>
Operator *CreateOp_CuDNNv4(BatchNormParam param);


#if DMLC_USE_CXX11
class CuDNNBatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));

    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CuDNNBatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "CuDNNBatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[cudnnbatchnorm::kOut],
            out_data[cudnnbatchnorm::kMean],
            out_data[cudnnbatchnorm::kInvVar],
            in_data[cudnnbatchnorm::kData],
            in_data[cudnnbatchnorm::kGamma],
            in_data[cudnnbatchnorm::kBeta]
           };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "inv_var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_inv_var"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  BatchNormParam param_;
};  // class CuDNNBatchNormProp

#endif  // DMLC_USE_CXX11
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUDNN_BATCH_NORM_INL_H_
