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
    .describe("If true AND the sliced dimension becomes 1, squeeze that dimension.");
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
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3> > outputs(size_);
    Tensor<xpu, 3> data;
    size_t leading = 1, trailing = 1;
    size_t mid = in_data[slice_enum::kData].shape_[axis_];
    for (int i = 0; i < axis_; ++i) {
      leading *= in_data[slice_enum::kData].shape_[i];
    }
    for (int i = axis_ + 1; i < in_data[slice_enum::kData].ndim(); ++i) {
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
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3> > grad_out(size_);
    Tensor<xpu, 3> grad;
    size_t leading = 1, trailing = 1;
    size_t mid = in_grad[slice_enum::kData].shape_[axis_];
    for (int i = 0; i < axis_; ++i) {
      leading *= in_grad[slice_enum::kData].shape_[i];
    }
    for (int i = axis_ + 1; i < in_grad[slice_enum::kData].ndim(); ++i) {
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
      ret.push_back(std::string("output") + static_cast<char>('0' + i));
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
    CHECK_EQ(in_shape->size(), 1);
    TShape dshape = in_shape->at(slice_enum::kData);
    if (dshape.ndim() == 0) return false;
    CHECK_GE(dshape.ndim(), static_cast<size_t>(param_.axis));
    CHECK_EQ(dshape[param_.axis] % param_.num_outputs, 0)
      << "num_outputs (" << param_.num_outputs
      << ") does not divide input dimension "
      << param_.axis << " (" << dshape[param_.axis] << ").";
    dshape[param_.axis] /= param_.num_outputs;
    if (param_.squeeze_axis && dshape[param_.axis] == 1) {
      for (int d = param_.axis; d < static_cast<int>(dshape.ndim()) - 1; ++d) {
        dshape[d] = dshape[d+1];
      }
      dshape = TShape(&dshape[0], &dshape[dshape.ndim()-1]);
    }
    out_shape->clear();
    for (int i = 0; i < param_.num_outputs; ++i) {
      out_shape->push_back(dshape);
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
