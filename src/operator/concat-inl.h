/*!
 * Copyright (c) 2015 by Contributors
 * \file concat-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CONCAT_INL_H_
#define MXNET_OPERATOR_CONCAT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./channel_op_common.h"

namespace mxnet {
namespace op {

namespace concat_enum {
enum ConcatOpInputs {kData0, kData1, kData2, kData3, kData4};
enum ConcatOpOutputs {kOut};
}  // namespace concat_enum

struct ConcatParam : public dmlc::Parameter<ConcatParam> {
  int num_args;
  int dim;
  DMLC_DECLARE_PARAMETER(ConcatParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be concated.");
    DMLC_DECLARE_FIELD(dim).set_range(0,  4).set_default(1)
    .describe("the dimension to be concated.");
  }
};  // struct ConcatParam

template<typename xpu>
class ConcatOp : public Operator {
 public:
  explicit ConcatOp(ConcatParam param)
    : size_(param.num_args), dimension_(param.dim) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req[concat_enum::kOut], kWriteTo);
    CHECK_LT(dimension_, in_data[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4> > data(size_);
    Tensor<xpu, 4> out;
    if (in_data[concat_enum::kData0].ndim() < 4) {
      uint32_t dim = 0;
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape;
        if (in_data[concat_enum::kData0].ndim() == 2)
          dshape = Shape4(in_data[i].shape_[0], in_data[i].shape_[1], 1, 1);
        else
          dshape = Shape4(in_data[i].shape_[0], in_data[i].shape_[1], in_data[i].shape_[2], 1);
        data[i] = in_data[i].get_with_shape<xpu, 4, real_t>(dshape, s);
        dim += in_data[i].shape_[dimension_];
      }
      Shape<4> dshape_out;
      int a, b, c;
      a = (dimension_ == 0) ? dim : in_data[concat_enum::kData0].shape_[0];
      b = (dimension_ == 1) ? dim : in_data[concat_enum::kData0].shape_[1];
      int dim2 = (in_data[concat_enum::kData0].ndim() == 2) ? 1
                        : in_data[concat_enum::kData0].shape_[2];
      c = (dimension_ == 2) ? dim : dim2;
      dshape_out = Shape4(a, b, c, 1);
      out = out_data[concat_enum::kOut].get_with_shape<xpu, 4, real_t>(dshape_out, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        data[i] = in_data[i].get<xpu, 4, real_t>(s);
      }
      out = out_data[concat_enum::kOut].get<xpu, 4, real_t>(s);
    }
    Concatenate(data, &out, dimension_);
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
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4> > grad_in(size_);
    Tensor<xpu, 4> grad;
    std::vector<bool> mask(size_, true);
    if (out_grad[concat_enum::kOut].ndim() < 4) {
      uint32_t dim = 0;
      for (int i = 0; i < size_; ++i) {
        Shape<4> dshape;
        if (out_grad[concat_enum::kOut].ndim() == 2)
          dshape = Shape4(in_grad[i].shape_[0], in_grad[i].shape_[1], 1, 1);
        else
          dshape = Shape4(in_grad[i].shape_[0], in_grad[i].shape_[1], in_grad[i].shape_[2], 1);
        dim += in_grad[i].shape_[dimension_];
        if (req[i] == kNullOp) {
          // Input doesn't need a gradient, don't propagate any
          mask[i] = false;
          // set the dimension so that Split knows how much to advance
          grad_in[i].shape_[dimension_] = dshape[dimension_];
          continue;
        }
        grad_in[i] = in_grad[i].get_with_shape<xpu, 4, real_t>(dshape, s);
        CHECK_EQ(req[i], kWriteTo);
      }
      Shape<4> dshape_out;
      int a, b, c;
      a = (dimension_ == 0) ? dim : in_grad[concat_enum::kData0].shape_[0];
      b = (dimension_ == 1) ? dim : in_grad[concat_enum::kData0].shape_[1];
      int dim2 = (out_grad[concat_enum::kOut].ndim() == 2) ? 1
                      : in_grad[concat_enum::kData0].shape_[2];
      c = (dimension_ == 2) ? dim : dim2;
      dshape_out = Shape4(a, b, c, 1);
      grad = out_grad[concat_enum::kOut].get_with_shape<xpu, 4, real_t>(dshape_out, s);
    } else {
      for (int i = 0; i < size_; ++i) {
        if (req[i] == kNullOp) {
          // Input doesn't need a gradient, don't propagate any
          mask[i] = false;
          // set the dimension so that Split knows how much to advance
          grad_in[i].shape_[dimension_] = in_grad[i].shape_[dimension_];
          continue;
        }
        grad_in[i] = in_grad[i].get<xpu, 4, real_t>(s);
        CHECK_EQ(req[i], kWriteTo);
      }
      grad = out_grad[concat_enum::kOut].get<xpu, 4, real_t>(s);
    }
    Split(grad, &grad_in, dimension_, mask);
  }

 private:
  int size_;
  int dimension_;
};  // class ConcatOp

template<typename xpu>
Operator *CreateOp(ConcatParam param);

#if DMLC_USE_CXX11
class ConcatProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    TShape dshape = in_shape->at(concat_enum::kData0);
    if (dshape.ndim() == 0) return false;
    CHECK_GT(dshape.ndim(), 1);
    CHECK_LT(static_cast<index_t>(param_.dim), dshape.ndim())
        <<"the dimension to be concated is not in the range of input's dimension";
    for (int i = 1; i < param_.num_args; ++i) {
      const TShape &tmp = in_shape->at(i);
      if (tmp.ndim() == 0) return false;
      for (index_t j = 0; j < dshape.ndim(); ++j) {
        if (j == static_cast<index_t>(param_.dim)) {
          dshape[param_.dim] += tmp[param_.dim];
        } else {
          CHECK_EQ(dshape[j], tmp[j])
              << "Incorrect shape[" << i << "]: "
              << tmp << ". "
              << "(first input shape: "
              << dshape << ")";
        }
      }
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ConcatProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Concat";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ConcatParam param_;
};  // class ConcatProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONCAT_INL_H_
