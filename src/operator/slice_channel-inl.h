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
  DMLC_DECLARE_PARAMETER(SliceChannelParam) {
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("Number of outputs to be sliced.");
  }
};  // struct SliceChannelParam

template<typename xpu>
class SliceChannelOp : public Operator {
 public:
  explicit SliceChannelOp(SliceChannelParam param)
    : size_(param.num_outputs) {}

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
    std::vector<Tensor<xpu, 4> > outputs(size_);
    Tensor<xpu, 4> data;
    if (in_data[slice_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[slice_enum::kData].shape_[0],
                               in_data[slice_enum::kData].shape_[1], 1, 1);
      data = in_data[slice_enum::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      Shape<4> slice_shape = dshape;
      slice_shape[1] = dshape[1] / size_;
      for (int i = 0; i < size_; ++i) {
        outputs[i] = out_data[i].get_with_shape<xpu, 4, real_t>(slice_shape, s);
      }
    } else {
      data = in_data[slice_enum::kData].get<xpu, 4, real_t>(s);
      for (int i = 0; i < size_; ++i) {
        outputs[i] = out_data[i].get<xpu, 4, real_t>(s);
      }
    }
    Split(data, &outputs, 1);
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
    std::vector<Tensor<xpu, 4> > grad_out(size_);
    Tensor<xpu, 4> grad;
    if (out_grad[slice_enum::kOut0].ndim() == 2) {
      Shape<4> slice_shape = Shape4(out_grad[slice_enum::kOut0].shape_[0],
                                    out_grad[slice_enum::kOut0].shape_[1], 1, 1);
      for (int i = 0; i < size_; ++i) {
        grad_out[i] = out_grad[i].get_with_shape<xpu, 4, real_t>(slice_shape, s);
      }
      Shape<4> dshape = slice_shape;
      dshape[1] *= size_;
      grad = in_grad[slice_enum::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        grad_out[i] = out_grad[i].get<xpu, 4, real_t>(s);
      }
      grad = in_grad[slice_enum::kData].get<xpu, 4, real_t>(s);
    }
    Concatenate(grad_out, &grad, 1);
  }

 private:
  int size_;
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
    CHECK_GT(dshape.ndim(), 1);
    CHECK_EQ(dshape[1] % param_.num_outputs, 0)
      << "Channel must be divided by the output number: "
      << dshape[1] << " / " << param_.num_outputs;
    dshape[1] /= param_.num_outputs;
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
