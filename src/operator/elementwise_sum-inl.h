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
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace elemsum {
enum ElementWiseSumOpInputs {kData0, kData1, kData2, kData3};
enum ElementWiseSumOpOutputs {kOut};
}  // namespace elemsum

struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be summed.");
  }
};

template<typename xpu, typename DType>
class ElementWiseSumOp : public Operator {
 public:
  explicit ElementWiseSumOp(ElementWiseSumParam param)
    : size_(param.num_args) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1);
    if (req[elemsum::kOut] == kNullOp) return;

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> out = out_data[elemsum::kOut].FlatTo2D<xpu, DType>(s);
    switch (size_) {
      case 2: {
        Tensor<xpu, 2, DType> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = in_data[elemsum::kData1].FlatTo2D<xpu, DType>(s);
        Assign(out, req[elemsum::kOut], in_0 + in_1);
        break;
      }
      case 3: {
        Tensor<xpu, 2, DType> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = in_data[elemsum::kData1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = in_data[elemsum::kData2].FlatTo2D<xpu, DType>(s);
        Assign(out, req[elemsum::kOut], in_0 + in_1 + in_2);
        break;
      }
      case 4: {
        Tensor<xpu, 2, DType> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = in_data[elemsum::kData1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = in_data[elemsum::kData2].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_3 = in_data[elemsum::kData3].FlatTo2D<xpu, DType>(s);
        Assign(out, req[elemsum::kOut], in_0 + in_1 + in_2 + in_3);
        break;
      }
      default: {
        Tensor<xpu, 2, DType> in_0 = in_data[elemsum::kData0].FlatTo2D<xpu, DType>(s);
        Assign(out, req[elemsum::kOut], F<mshadow_op::identity>(in_0));
        for (int i = 1; i < size_; ++i) {
          out += in_data[i].FlatTo2D<xpu, DType>(s);
        }
        break;
      }
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
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> ograd = out_grad[elemsum::kOut].FlatTo2D<xpu, DType>(s);
    for (int i = 0; i < size_; ++i) {
      if (req[i] == kNullOp || req[i] == kWriteInplace) continue;
      Tensor<xpu, 2, DType> igrad = in_grad[i].FlatTo2D<xpu, DType>(s);
      Assign(igrad, req[i], F<mshadow_op::identity>(ograd));
    }
  }
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("size_", size_);
    writer->EndObject();
  }
  inline void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("size_", &size_);
    helper.ReadAllFields(reader);
  }

 private:
  int size_;
};  // class ElementWiseSumOp

template<typename xpu>
Operator* CreateOp(ElementWiseSumParam param, int dtype);

#if DMLC_USE_CXX11
class ElementWiseSumProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    int sidx = -1;
    for (int i = 0; i < param_.num_args; ++i) {
      if (in_shape->at(i).ndim() != 0) {
        sidx = i;
        break;
      }
    }
    if (sidx == -1) return false;
    for (int i = 0; i < param_.num_args; ++i) {
      if (i != sidx) {
        SHAPE_ASSIGN_CHECK(*in_shape, i, in_shape->at(sidx));
      }
    }
    out_shape->clear();
    out_shape->push_back(in_shape->at(sidx));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    size_t nin = in_type->size();
    CHECK_EQ(nin, static_cast<size_t>(param_.num_args));

    int dtype = -1;
    for (size_t i = 0; i < nin; ++i) {
      if (dtype == -1) {
        dtype = in_type->at(i);
      } else {
        CHECK(in_type->at(i) == dtype ||
              in_type->at(i) == -1) <<
              "This operator requires uniform type";
      }
    }

    if (dtype == -1) {
      LOG(FATAL) << "At least one input type needs to be known";
      return false;
    }

    in_type->clear();
    for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ElementWiseSumProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "ElementWiseSum";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[0], in_grad[0]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[0], out_data[0]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ElementWiseSumParam param_;
};  // class ElementWiseSumProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_SUM_INL_H_
