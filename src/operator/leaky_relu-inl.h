/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu-inl.h
 * \brief leaky relu family operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LEAKY_RELU_INL_H_
#define MXNET_OPERATOR_LEAKY_RELU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace leakyrelu {
enum LeakyReLUOpInputs {kData, kGamma};
enum LeakyReLUOpOutputs {kOut, kMask};
enum LeakyReLUOpType {kLeakyReLU, kPReLU, kRReLU, kELU};
enum LeakyReLUOpResource {kRandom};
}  // namespace leakyrelu

struct LeakyReLUParam : public dmlc::Parameter<LeakyReLUParam> {
  // use int for enumeration
  int act_type;
  float slope;
  float lower_bound;
  float upper_bound;
  DMLC_DECLARE_PARAMETER(LeakyReLUParam) {
    DMLC_DECLARE_FIELD(act_type).set_default(leakyrelu::kLeakyReLU)
    .add_enum("rrelu", leakyrelu::kRReLU)
    .add_enum("leaky", leakyrelu::kLeakyReLU)
    .add_enum("prelu", leakyrelu::kPReLU)
    .add_enum("elu", leakyrelu::kELU)
    .describe("Activation function to be applied.");
    DMLC_DECLARE_FIELD(slope).set_default(0.25f)
    .describe("Init slope for the activation. (For leaky and elu only)");
    DMLC_DECLARE_FIELD(lower_bound).set_default(0.125f)
    .describe("Lower bound of random slope. (For rrelu only)");
    DMLC_DECLARE_FIELD(upper_bound).set_default(0.334f)
    .describe("Upper bound of random slope. (For rrelu only)");
  }
};

struct prelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 0.0f : a;
  }
};

template<typename xpu>
class LeakyReLUOp : public Operator {
 public:
  explicit LeakyReLUOp(LeakyReLUParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    Tensor<xpu, 4> mask;
    Tensor<xpu, 1> weight;
    if (in_data[leakyrelu::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[leakyrelu::kData].shape_[0],
                               in_data[leakyrelu::kData].shape_[1], 1, 1);
      data = in_data[leakyrelu::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[leakyrelu::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
    } else {
      data = in_data[leakyrelu::kData].get<xpu, 4, real_t>(s);
      out = out_data[leakyrelu::kOut].get<xpu, 4, real_t>(s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get<xpu, 4, real_t>(s);
      }
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        Assign(out, req[leakyrelu::kOut], F<mshadow_op::xelu>(data, param_.slope));
        break;
      }
      case leakyrelu::kPReLU: {
        weight = in_data[leakyrelu::kGamma].get<xpu, 1, real_t>(s);
        Assign(out, req[leakyrelu::kOut],
               F<mshadow_op::xelu>(data, broadcast<1>(weight, out.shape_)));
        break;
      }
      case leakyrelu::kRReLU: {
        if (ctx.is_train) {
          Random<xpu>* prnd = ctx.requested[leakyrelu::kRandom].get_random<xpu, real_t>(s);
          mask = prnd->uniform(mask.shape_);
          mask = mask * (param_.upper_bound - param_.lower_bound) + param_.lower_bound;
          Assign(out, req[leakyrelu::kOut], F<mshadow_op::xelu>(data, mask));
        } else {
          const float slope = (param_.lower_bound + param_.upper_bound) / 2.0f;
          Assign(out, req[leakyrelu::kOut], F<mshadow_op::xelu>(data, slope));
        }
        break;
      }
      case leakyrelu::kELU: {
        Assign(out, req[leakyrelu::kOut], F<mshadow_op::elu>(data, param_.slope));
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> output;
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> gdata;
    Tensor<xpu, 4> grad;
    Tensor<xpu, 4> mask;
    Tensor<xpu, 1> weight;
    Tensor<xpu, 1> grad_weight;
    if (out_grad[leakyrelu::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[leakyrelu::kOut].shape_[0],
                               out_grad[leakyrelu::kOut].shape_[1], 1, 1);
      grad = out_grad[leakyrelu::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      gdata = in_grad[leakyrelu::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      output = out_data[leakyrelu::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
      if (param_.act_type == leakyrelu::kPReLU) {
        data = in_data[leakyrelu::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
    } else {
      grad = out_grad[leakyrelu::kOut].get<xpu, 4, real_t>(s);
      gdata = in_grad[leakyrelu::kData].get<xpu, 4, real_t>(s);
      output = out_data[leakyrelu::kOut].get<xpu, 4, real_t>(s);
      if (param_.act_type == leakyrelu::kRReLU) {
        mask = out_data[leakyrelu::kMask].get<xpu, 4, real_t>(s);
      }
      if (param_.act_type == leakyrelu::kPReLU) {
        data = in_data[leakyrelu::kData].get<xpu, 4, real_t>(s);
      }
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::xelu_grad>(output, param_.slope) * grad);
        break;
      }
      case leakyrelu::kPReLU: {
        weight = in_data[leakyrelu::kGamma].get<xpu, 1, real_t>(s);
        grad_weight = in_grad[leakyrelu::kGamma].get<xpu, 1, real_t>(s);
        grad_weight = sumall_except_dim<1>(F<prelu_grad>(data) * grad);
        gdata = F<mshadow_op::xelu_grad>(data, broadcast<1>(weight, data.shape_)) * grad;
        break;
      }
      case leakyrelu::kRReLU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::xelu_grad>(output, mask) * grad);
        break;
      }
      case leakyrelu::kELU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::elu_grad>(output, param_.slope) * grad);
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

 private:
  LeakyReLUParam param_;
};  // class LeakyReLUOp

template<typename xpu>
Operator* CreateOp(LeakyReLUParam type);

#if DMLC_USE_CXX11
class LeakyReLUProp : public OperatorProperty {
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
    if (param_.act_type == leakyrelu::kPReLU) {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, gamma]";
    } else {
      CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    }
    const TShape &dshape = in_shape->at(leakyrelu::kData);
    if (dshape.ndim() == 0) return false;
    if (param_.act_type == leakyrelu::kPReLU) {
      in_shape->at(leakyrelu::kGamma) = TShape(Shape1(dshape[1]));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    if (param_.act_type == leakyrelu::kRReLU) {
      out_shape->push_back(dshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LeakyReLUProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LeakyReLU";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {out_grad[leakyrelu::kOut],
              out_data[leakyrelu::kOut],
              in_data[leakyrelu::kData],
              in_data[leakyrelu::kGamma]};
    } else if (param_.act_type == leakyrelu::kRReLU) {
      return {out_grad[leakyrelu::kOut], out_data[leakyrelu::kMask], out_data[leakyrelu::kOut]};
    } else {
      return {out_grad[leakyrelu::kOut], out_data[leakyrelu::kData]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[leakyrelu::kOut], in_grad[leakyrelu::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {};
    } else {
      return {{in_data[leakyrelu::kData], out_data[leakyrelu::kOut]}};
    }
  }

  std::vector<std::string> ListArguments() const override {
    if (param_.act_type == leakyrelu::kPReLU) {
      return {"data", "gamma"};
    } else {
      return {"data"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return {"output", "mask"};
    } else {
      return {"output"};
    }
  }

  int NumOutputs() const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    if (param_.act_type == leakyrelu::kRReLU) {
      return {ResourceRequest::kRandom};
    } else {
      return std::vector<ResourceRequest>();
    }
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  LeakyReLUParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_LEAKY_RELU_INL_H_

