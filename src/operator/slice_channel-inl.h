/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SLICE_CHANNEL_INL_H_
#define MXNET_OPERATOR_SLICE_CHANNEL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./channel_op_common.h"

namespace mxnet {
namespace op {

namespace slice_enum {
enum SliceChannelOpInputs {kData};
enum SliceChannelOpOutputs {kOut0, kOut1, kOut2, kOut3, kOut4};
}  // namespace slice_enum

struct SliceChannelParam : public dmlc::Parameter<SliceChannelParam> {
  int num_outputs;
  int axis;
  bool squeeze_axis;
  DMLC_DECLARE_PARAMETER(SliceChannelParam) {
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("Number of outputs to be sliced.");
    DMLC_DECLARE_FIELD(axis).set_default(1)
    .describe("Dimension along which to slice.");
    DMLC_DECLARE_FIELD(squeeze_axis).set_default(0)
    .describe("If true, the dimension will be squeezed."
              " Also, input.shape[axis] must be the same as `num_outputs`"
              " when squeeze_axis is turned on.");
  }
};  // struct SliceChannelParam

template<typename xpu>
class SliceChannelOp : public Operator {
 public:
  explicit SliceChannelOp(SliceChannelParam param)
    : size_(param.num_outputs), axis_(param.axis) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3> > outputs(size_);
    Tensor<xpu, 3> data;
    size_t leading = 1, trailing = 1;
    int real_axis = axis_;
    if (real_axis < 0) {
      real_axis += in_data[slice_enum::kData].ndim();
    }
    CHECK_LT(real_axis, in_data[slice_enum::kData].ndim());
    size_t mid = in_data[slice_enum::kData].shape_[real_axis];
    for (int i = 0; i < real_axis; ++i) {
      leading *= in_data[slice_enum::kData].shape_[i];
    }
    for (int i = real_axis + 1; i < in_data[slice_enum::kData].ndim(); ++i) {
      trailing *= in_data[slice_enum::kData].shape_[i];
    }
    Shape<3> dshape = Shape3(leading, mid, trailing);
    Shape<3> slice_shape = Shape3(leading, mid / size_, trailing);
    data = in_data[slice_enum::kData].get_with_shape<xpu, 3, real_t>(dshape, s);
    for (int i = 0; i < size_; ++i) {
      outputs[i] = out_data[i].get_with_shape<xpu, 3, real_t>(slice_shape, s);
    }
    Split(data, &outputs, 1, req);
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
    CHECK_EQ(out_grad.size(), static_cast<size_t>(size_));
    CHECK_EQ(in_grad.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3> > grad_out(size_);
    Tensor<xpu, 3> grad;
    size_t leading = 1, trailing = 1;
    int real_axis = axis_;
    if (real_axis < 0) {
        real_axis += in_grad[slice_enum::kData].ndim();
    }
    CHECK_LT(real_axis, in_grad[slice_enum::kData].ndim());
    size_t mid = in_grad[slice_enum::kData].shape_[real_axis];
    for (int i = 0; i < real_axis; ++i) {
      leading *= in_grad[slice_enum::kData].shape_[i];
    }
    for (int i = real_axis + 1; i < in_grad[slice_enum::kData].ndim(); ++i) {
      trailing *= in_grad[slice_enum::kData].shape_[i];
    }
    Shape<3> dshape = Shape3(leading, mid, trailing);
    Shape<3> slice_shape = Shape3(leading, mid / size_, trailing);
    grad = in_grad[slice_enum::kData].get_with_shape<xpu, 3, real_t>(dshape, s);
    for (int i = 0; i < size_; ++i) {
      grad_out[i] = out_grad[i].get_with_shape<xpu, 3, real_t>(slice_shape, s);
    }
    Concatenate(grad_out, &grad, 1, req[slice_enum::kData]);
  }

 private:
  int size_;
  int axis_;
};  // class SliceChannelOp


template<typename xpu>
Operator *CreateOp(SliceChannelParam param);


#if DMLC_USE_CXX11
class SliceChannelProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_outputs; ++i) {
      std::ostringstream os;
      os << "output" << i;
      ret.push_back(os.str());
    }
    return ret;
  }

  int NumOutputs() const override {
    return param_.num_outputs;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U);
    TShape dshape = in_shape->at(slice_enum::kData);
    TShape ishape = in_shape->at(slice_enum::kData);
    if (dshape.ndim() == 0) return false;
    if (param_.axis >= 0) {
      CHECK_LT(static_cast<size_t>(param_.axis), dshape.ndim());
    } else {
      CHECK_LT(param_.axis + dshape.ndim(), dshape.ndim());
    }
    int real_axis = param_.axis;
    if (real_axis < 0) {
      real_axis += dshape.ndim();
    }
    CHECK_EQ(dshape[real_axis] % param_.num_outputs, 0U)
      << "num_outputs (" << param_.num_outputs
      << ") does not divide input dimension "
      << real_axis << " (" << dshape[real_axis] << ").";
    if (param_.squeeze_axis && ishape[real_axis] != 0) {
      CHECK_EQ(ishape[real_axis], param_.num_outputs)
        << "If squeeze axis is True, the size of the sliced axis must be the same as num_outputs."
        << " Input shape=" << ishape << ", axis=" << real_axis
        << ", num_outputs=" << param_.num_outputs << ".";
    }
    dshape[real_axis] /= param_.num_outputs;
    if (param_.squeeze_axis && (dshape[real_axis] == 1 || ishape[real_axis] == 0)) {
      for (int d = real_axis; d < static_cast<int>(dshape.ndim()) - 1; ++d) {
        dshape[d] = dshape[d+1];
      }
      dshape = TShape(&dshape[0], &dshape[dshape.ndim()-1]);
    }
    CHECK_EQ((*out_shape).size(), param_.num_outputs) << "Size of output shape mismatch!";
    for (int i = 0; i < param_.num_outputs; ++i) {
      SHAPE_ASSIGN_CHECK(*out_shape, i, dshape);
      // Perform incomplete shape inference.
      // We can back-calculate the inshape based on the out_shape.
      TShape back_calculate_dshape = ishape;
      if (param_.squeeze_axis && (dshape.ndim() == ishape.ndim() - 1)) {
        for (int d = 0; d < real_axis; ++d) {
          back_calculate_dshape[d] = (*out_shape)[i][d];
        }
        back_calculate_dshape[real_axis] = param_.num_outputs;
        for (int d = real_axis + 1; d < static_cast<int>(ishape.ndim()); ++d) {
          back_calculate_dshape[d] = (*out_shape)[i][d - 1];
        }
      } else {
        for (int d = 0; d < static_cast<int>(ishape.ndim()); ++d) {
          back_calculate_dshape[d] = (*out_shape)[i][d];
          if (d == real_axis) {
            back_calculate_dshape[d] *= param_.num_outputs;
          }
        }
      }
      SHAPE_ASSIGN_CHECK(*in_shape, slice_enum::kData, back_calculate_dshape);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SliceChannelProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SliceChannel";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  SliceChannelParam param_;
};  // class SliceChannelProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SLICE_CHANNEL_INL_H_
