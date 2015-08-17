/*!
 * Copyright (c) 2015 by Contributors
 * \file elemementwise_sum-inl.h
 * \brief elementwise sum
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_

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

enum ElementWiseSumOpInputs {kData0, kData1, kData2, kData3};
enum ElementWiseSumOpOutputs {kOut};

struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int size;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(size).set_range(1, 100);
  }
};

template<typename xpu>
class ElementWiseSumOp : public Operator {
 public:
  explicit ElementWiseSumOp(ElementWiseSumParam param)
      : size_(param.size) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    if (req[kOut] == kNullOp) return;

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    switch (size_) {
      case 2: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0 + in_1);
        break;
      }
      case 3: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_2 = in_data[kData2].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0 + in_1 + in_2);
        break;
      }
      case 4: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_2 = in_data[kData2].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_3 = in_data[kData3].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0 + in_1 + in_2 + in_3);
        break;
      }
      default: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0);
        for (int i = 0; i < size_; ++i) {
          out += in_data[i].FlatTo2D<xpu, real_t>(s);
        }
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> ograd = out_grad[kOut].FlatTo2D<xpu, real_t>(s);

    for (int i = 0; i < size_; ++i) {
      if (req[i] == kNullOp || req[i] == kWriteInplace) continue;
      Tensor<xpu, 2> igrad = in_grad[i].FlatTo2D<xpu, real_t>(s);
      Assign(igrad, req[i], ograd);
    }
  }

 private:
  int size_;
};  // class ElementWiseSumOp

template<typename xpu>
Operator* CreateOp(ElementWiseSumParam param);

#if DMLC_USE_CXX11
class ElementWiseSumProp : public OperatorProperty {
 public:
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    // TODO(bing) change directly to vector of pairs begin end
    std::map<std::string, std::string> kmap(kwargs.begin(), kwargs.end());
    param_.Init(kmap);
  }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.size));
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    for (int i = 1; i < param_.size; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, dshape);
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  virtual OperatorProperty* Copy() const {
    auto ptr = new ElementWiseSumProp();
    ptr->param_ = param_;
    return ptr;
  }

  virtual std::string TypeString() const {
    return "ElementWiseSum";
  }

  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return out_grad;
  }

  virtual std::vector<std::pair<int, int> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<int> &in_grad) const {
    return {{out_grad[0], in_grad[0]}};
  }

  virtual std::vector<std::pair<int, int> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {{in_data[0], out_data[0]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  ElementWiseSumParam param_;
};  // class ElementWiseSumProp

#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_
