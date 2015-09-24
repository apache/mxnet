/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op-inl.h
 * \brief Elementwise binary operation, plus, minus, mul, div
*/
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

enum ElementWiseBinaryOpInputs {kLhs, kRhs};
enum ElementWiseBinaryOpOutputs {kOut};
enum ElementWiseBinaryOpType {kPlus, kMinus, kMul, kDiv};

template<typename Op>
inline ElementWiseBinaryOpType GetOpType();

template<typename Op>
inline const char* GetOpTypeString();

template<>
inline ElementWiseBinaryOpType GetOpType<mshadow::op::plus>() {
  return kPlus;
}
template<>
inline ElementWiseBinaryOpType GetOpType<mshadow::op::minus>() {
  return kMinus;
}
template<>
inline ElementWiseBinaryOpType GetOpType<mshadow::op::mul>() {
  return kMul;
}
template<>
inline ElementWiseBinaryOpType GetOpType<mshadow::op::div>() {
  return kDiv;
}

template<>
inline const char* GetOpTypeString<mshadow::op::plus>() {
  return "_Plus";
}
template<>
inline const char* GetOpTypeString<mshadow::op::minus>() {
  return "_Minus";
}

template<>
inline const char* GetOpTypeString<mshadow::op::mul>() {
  return "_Mul";
}

template<>
inline const char* GetOpTypeString<mshadow::op::div>() {
  return "_Div";
}

template<typename xpu, typename ForwardOp>
class ElementWiseBinaryOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> lhs = in_data[kLhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> rhs = in_data[kRhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[kOut], F<ForwardOp>(lhs, rhs));
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
    CHECK(in_data.size() == 2 && in_grad.size() == 2);
    CHECK_EQ(req.size(), 2);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> m_out_grad = out_grad[kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> lhs_grad = in_grad[kLhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> rhs_grad = in_grad[kRhs].FlatTo2D<xpu, real_t>(s);
    switch (GetOpType<ForwardOp>()) {
      case kPlus: {
        Assign(lhs_grad, req[kLhs], F<mshadow_op::identity>(m_out_grad));
        Assign(rhs_grad, req[kRhs], F<mshadow_op::identity>(m_out_grad));
        break;
      }
      case kMinus: {
        Assign(lhs_grad, req[kLhs], F<mshadow_op::identity>(m_out_grad));
        Assign(rhs_grad, req[kRhs], F<mshadow_op::negation>(m_out_grad));
        break;
      }
      case kMul: {
        Tensor<xpu, 2> lhs_data = in_data[kLhs].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> rhs_data = in_data[kRhs].FlatTo2D<xpu, real_t>(s);
        // rhs cannot do inplace
        CHECK_NE(req[kRhs], kWriteInplace);
        Assign(rhs_grad, req[kRhs], lhs_data * m_out_grad);
        Assign(lhs_grad, req[kLhs], rhs_data * m_out_grad);
        break;
      }
      case kDiv: {
        Tensor<xpu, 2> lhs_data = in_data[kLhs].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> rhs_data = in_data[kRhs].FlatTo2D<xpu, real_t>(s);
        // rhs cannot do inplace
        CHECK_NE(req[kRhs], kWriteInplace);
        Assign(rhs_grad, req[kRhs],
               F<mshadow_op::negation>(m_out_grad * lhs_data) / F<mshadow_op::square>(rhs_data));
        Assign(lhs_grad, req[kLhs], m_out_grad / rhs_data);
        break;
      }
    }
  }
};  // class ElementWiseBinaryOp


template<typename xpu>
inline Operator* CreateElementWiseBinaryOp_(ElementWiseBinaryOpType type) {
  switch (type) {
    case kPlus:
      return new ElementWiseBinaryOp<xpu, mshadow::op::plus>();
    case kMinus:
      return new ElementWiseBinaryOp<xpu, mshadow::op::minus>();
    case kMul:
      return new ElementWiseBinaryOp<xpu, mshadow::op::mul>();
    case kDiv:
      return new ElementWiseBinaryOp<xpu, mshadow::op::div>();
  }
  LOG(FATAL) << "uknown op type";
  return NULL;
}

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateElementWiseBinaryOp(ElementWiseBinaryOpType type);

#if DMLC_USE_CXX11
template<typename ForwardOp>
class ElementWiseBinaryOpProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    CHECK_EQ(kwargs.size(), 0)
        << TypeString() << " do not take any additional keyword arguments besides lhs and rhs";
  }
  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[lhs, rhs]";
    if (in_shape->at(kLhs).ndim() != 0) {
      SHAPE_ASSIGN_CHECK(*in_shape, kRhs, in_shape->at(kLhs));
    } else if (in_shape->at(kRhs).ndim() != 0) {
      in_shape->at(kLhs) = in_shape->at(kRhs);
    } else {
      return false;
    }
    const TShape &dshape = in_shape->at(kLhs);
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  std::vector<std::string> ListArguments() const override {
    return {"lhs", "rhs"};
  }

  OperatorProperty* Copy() const override {
    return new ElementWiseBinaryOpProp<ForwardOp>();
  }

  std::string TypeString() const override {
    return GetOpTypeString<ForwardOp>();
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    switch (GetOpType<ForwardOp>()) {
      case kPlus:
      case kMinus:
        return {out_grad[kOut]};
      case kMul:
      case kDiv:
        return {out_grad[kOut], in_data[kLhs], in_data[kRhs]};
    }
    LOG(FATAL) << "not reached";
    return {};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    switch (GetOpType<ForwardOp>()) {
      case kPlus:
      case kMinus:
        return {};
      case kMul:
      case kDiv:
        return {{out_grad[kOut], in_grad[kLhs]}};
    }
    LOG(FATAL) << "not reached";
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kLhs], out_data[kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
