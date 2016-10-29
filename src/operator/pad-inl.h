/*!
 * Copyright (c) 2016 by Contributors
 * \file pad-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/

#ifndef MXNET_OPERATOR_PAD_INL_H_
#define MXNET_OPERATOR_PAD_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace pad_enum {
enum PadOpInputs { kData };
enum PadOpType { kConstant, kReplicate };
enum PadOpOutputs { kOut };
}

struct PadParam : public dmlc::Parameter<PadParam> {
  int pad_type;
  double padding_constant;
  TShape pad_shape;
  DMLC_DECLARE_PARAMETER(PadParam) {
    DMLC_DECLARE_FIELD(pad_type)
        .add_enum("constant", pad_enum::kConstant)
        .add_enum("replicate", pad_enum::kReplicate)
        .describe(
            "Padding type to use. \"constant\" pads all values with a constant "
            "value,"
            "the value of which can be specified with the padding_constant"
            "option. \"replicate\" uses the boundary values of the array as "
            "padding.");

    DMLC_DECLARE_FIELD(pad_shape).describe(
        "A tuple of padding sizes of length 2*r, where r is the rank of the "
        "input tensor. ");
    DMLC_DECLARE_FIELD(padding_constant)
        .describe(
            "This option is only used when pad_type is \"constant\". This "
            "value will be used as the padding value. Defaults to 0 if not "
            "specified.")
        .set_default(0.0);
  }
};

template <typename xpu, typename DType>
class PadOp : public Operator {
 public:
  explicit PadOp(PadParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Get any size input + output into required form
    int rank = in_data[pad_enum::kData].ndim();
    auto pad = param_.pad_shape;
    DType padding_constant = param_.padding_constant;

    if ((rank == 4) && !pad[0] && !pad[1] && !pad[2] && !pad[3]) {
      Tensor<xpu, 4, DType> data =
          in_data[pad_enum::kData].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out =
          out_data[pad_enum::kOut].get<xpu, 4, DType>(s);
      pad_image_2d(out, data, param_.pad_shape, param_.pad_type,
                   padding_constant);
    } else {
      LOG(FATAL) << "Only 4d input tensors and padding applied to the last "
                    "two dimensions is currently implemented. ";
    }

    // Assign(out, req[pad_enum::kOut], F<mshadow_op::identity>(data));
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Get any size input + output into required form
    auto pad = param_.pad_shape;
    int rank = in_grad[pad_enum::kData].ndim();
    // Currently only support rank 4
    if ((rank == 4) && !pad[0] && !pad[1] && !pad[2] && !pad[3]) {
      Tensor<xpu, 4, DType> in = in_grad[pad_enum::kData].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out =
          out_grad[pad_enum::kOut].get<xpu, 4, DType>(s);
      if (req[pad_enum::kData] == kWriteTo) in = 0.0f;

      pad_image_2d_grad(in, out, param_.pad_shape, param_.pad_type);
    } else {
      LOG(FATAL) << "Only 4d input tensors and padding applied to the last "
                    "two dimensions is currently implemented. ";
    }
  }

 private:
  PadParam param_;
};  // class PadOp

template <typename xpu>
Operator *CreateOp(PadParam param, int dtype);

#if DMLC_USE_CXX11
class PadProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 1; }

  std::vector<std::string> ListArguments() const override { return {"data"}; }

  std::vector<std::string> ListOutputs() const override { return {"output"}; }

  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Can only be one input to symbol.";

    const TShape &dshape = (*in_shape)[pad_enum::kData];
    if (dshape.ndim() == 0) return false;
    TShape oshape = dshape;
    for (int i = 0; i < dshape.ndim(); ++i) {
      oshape[i] =
          param_.pad_shape[2 * i] + param_.pad_shape[2 * i + 1] + dshape[i];
    }
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new PadProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "Pad"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[pad_enum::kOut]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  PadParam param_;
};      // class PadProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_MASK_INL_H_
