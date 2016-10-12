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

template<typename xpu, typename DType>
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
    CHECK_LT(dimension_, in_data[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3, DType> > data(size_);
    Tensor<xpu, 3, DType> out;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < dimension_; ++i) {
      leading *= out_data[concat_enum::kOut].shape_[i];
    }
    for (int i = dimension_ + 1; i < out_data[concat_enum::kOut].ndim(); ++i) {
      trailing *= out_data[concat_enum::kOut].shape_[i];
    }
    size_t mid = out_data[concat_enum::kOut].shape_[dimension_];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    out = out_data[concat_enum::kOut].get_with_shape<xpu, 3, DType>(oshape, s);

    for (int i = 0; i < size_; ++i) {
      Shape<3> dshape = Shape3(leading, in_data[i].shape_[dimension_], trailing);
      data[i] = in_data[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Concatenate(data, &out, 1, req[concat_enum::kOut]);
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
    std::vector<Tensor<xpu, 3, DType> > grad_in(size_);
    Tensor<xpu, 3, DType> grad;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < dimension_; ++i) {
      leading *= out_grad[concat_enum::kOut].shape_[i];
    }
    for (int i = dimension_ + 1; i < out_grad[concat_enum::kOut].ndim(); ++i) {
      trailing *= out_grad[concat_enum::kOut].shape_[i];
    }
    size_t mid = out_grad[concat_enum::kOut].shape_[dimension_];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    grad = out_grad[concat_enum::kOut].get_with_shape<xpu, 3, DType>(oshape, s);

    for (int i = 0; i < size_; ++i) {
      Shape<3> dshape = Shape3(leading, in_grad[i].shape_[dimension_], trailing);
      grad_in[i] = in_grad[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Split(grad, &grad_in, 1, req);
  }

 private:
  int size_;
  int dimension_;
};  // class ConcatOp

template<typename xpu>
Operator *CreateOp(ConcatParam param, int dtype);

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
    CHECK_GE(dshape.ndim(), 1);
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

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = -1;

    for (size_t i = 0; i < in_type->size(); ++i) {
      if (dtype == -1) {
        dtype = in_type->at(i);
      } else {
        CHECK(in_type->at(i) == dtype ||
              in_type->at(i) == -1) <<
              "Non-uniform data type in Concat";
      }
    }

    if (dtype == -1) {
      LOG(FATAL) << "Not enough information to infer type in Concat.";
      return false;
    }

    size_t nin = this->ListArguments().size();
    in_type->clear();
    for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);

    size_t naux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (size_t i = 0; i < naux; ++i) aux_type->push_back(dtype);

    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);

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

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ConcatParam param_;
};  // class ConcatProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONCAT_INL_H_
