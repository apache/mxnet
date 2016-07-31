/*!
 * Copyright (c) 2015 by Contributors
 * \file cast-inl.h
 * \brief cast operator
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_CAST_INL_H_
#define MXNET_OPERATOR_CAST_INL_H_

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
namespace cast {
enum CastOpInputs {kData};
enum CastOpOutputs {kOut};
}  // cast

struct CastParam : public dmlc::Parameter<CastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Target data type.");
  }
};

/**
 * \brief This is the implementation of cast operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename SrcDType, typename DstDType>
class CastOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, SrcDType> data = in_data[cast::kData].FlatTo2D<xpu, SrcDType>(s);
    Tensor<xpu, 2, DstDType> out = out_data[cast::kOut].FlatTo2D<xpu, DstDType>(s);
    Assign(out, req[cast::kOut], tcast<DstDType>(data));
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
    CHECK_EQ(in_grad.size(), 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DstDType> m_out_grad = out_grad[cast::kOut].FlatTo2D<xpu, DstDType>(s);
    Tensor<xpu, 2, SrcDType> m_in_grad = in_grad[cast::kData].FlatTo2D<xpu, SrcDType>(s);
    Assign(m_in_grad, req[cast::kData], tcast<SrcDType>(m_out_grad));
  }
};  // class CastOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CastParam param, std::vector<int> *in_type);

#if DMLC_USE_CXX11
class CastProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(cast::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    out_type->clear();
    out_type->push_back(param_.dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CastProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Cast";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[cast::kOut]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CastParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CAST_INL_H_
