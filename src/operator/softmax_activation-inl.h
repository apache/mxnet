/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_activation-inl.h
 * \brief SoftmaxActivation operator
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_SOFTMAX_ACTIVATION_INL_H_
#define MXNET_OPERATOR_SOFTMAX_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace softmax_activation {
enum SoftmaxActivationOpInputs {kData};
enum SoftmaxActivationOpOutputs {kOut};
enum SoftmaxActivationOpType {kInstance, kChannel};
enum SoftmaxActivationOpResource {kTempSpace};
}  // softmax_activation

struct SoftmaxActivationParam : public dmlc::Parameter<SoftmaxActivationParam> {
  // use int for enumeration
  int mode;
  DMLC_DECLARE_PARAMETER(SoftmaxActivationParam) {
    DMLC_DECLARE_FIELD(mode)
    .add_enum("instance", softmax_activation::kInstance)
    .add_enum("channel", softmax_activation::kChannel)
    .set_default(softmax_activation::kInstance)
    .describe("Specifies how to compute the softmax. If set to ``instance``, "
              "it computes softmax for each instance. If set to ``channel``, "
              "It computes cross channel softmax for each position of each instance.");
  }
};

/**
 * \brief This is the implementation of softmax_activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class SoftmaxActivationOp : public Operator {
 public:
  explicit SoftmaxActivationOp(SoftmaxActivationParam p) {
    this->param_ = p;
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.mode == softmax_activation::kInstance) {
      Tensor<xpu, 2> data = in_data[softmax_activation::kData].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> out = out_data[softmax_activation::kOut].FlatTo2D<xpu, real_t>(s);
      Softmax(out, data);
    } else {
      CHECK_GE(in_data[softmax_activation::kData].ndim(), 3)
        << "Input need to have a least 3 dimensions when mode=channel";
      int n = in_data[softmax_activation::kData].size(0);
      int k = in_data[softmax_activation::kData].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[softmax_activation::kData].Size()/n/k));
      Tensor<xpu, 3, real_t> data =
        in_data[softmax_activation::kData].get_with_shape<xpu, 3, real_t>(s3, s);
      Tensor<xpu, 3, real_t> out =
        out_data[softmax_activation::kOut].get_with_shape<xpu, 3, real_t>(s3, s);
      Softmax(out, data);
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
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1U);
    // Use 3d tensor for both mode -> {instance, channel}. Get shapes
    int total_size = in_grad[softmax_activation::kData].Size();
    int batch_size = in_grad[softmax_activation::kData].shape_[0];
    int channel_num = in_grad[softmax_activation::kData].shape_[1];
    int rest_size = total_size / (batch_size * channel_num);
    const Shape<3> data_shape = Shape3(batch_size, channel_num, rest_size);
    // Get tensors
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> m_out_grad =
      out_grad[softmax_activation::kOut].get_with_shape<xpu, 3, real_t>(data_shape, s);
    Tensor<xpu, 3> m_out_data =
      out_data[softmax_activation::kOut].get_with_shape<xpu, 3, real_t>(data_shape, s);
    Tensor<xpu, 3> m_in_grad =
      in_grad[softmax_activation::kData].get_with_shape<xpu, 3, real_t>(data_shape, s);
    // get requested temp space
    Tensor<xpu, 2> workspace = ctx.requested[softmax_activation::kTempSpace].get_space<xpu>(
        Shape2(batch_size, rest_size), s);
    workspace = reduce_with_axis<red::sum, false>(m_out_grad * m_out_data, 1);
    Assign(m_in_grad, req[softmax_activation::kData],
        m_out_data * (m_out_grad - broadcast_with_axis(workspace, 0, channel_num)));
  }

 private:
  SoftmaxActivationParam param_;
};  // class SoftmaxActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxActivationParam type);

#if DMLC_USE_CXX11
class SoftmaxActivationProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const TShape &dshape = in_shape->at(softmax_activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxActivation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[softmax_activation::kOut], out_data[softmax_activation::kOut]};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[softmax_activation::kOut], in_grad[softmax_activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmax_activation::kData], out_data[softmax_activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  SoftmaxActivationParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_ACTIVATION_INL_H_
