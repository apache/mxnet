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

namespace elembinary {
enum ElementWiseBinaryOpInputs {kLhs, kRhs};
enum ElementWiseBinaryOpOutputs {kOut};
enum ElementWiseBinaryOpType {kPlus, kMinus, kMul, kDiv, kPower, kMaximum, kMinimum};
enum ElementWiseBinaryOpResource { kTempSpace };
}  // elembinary

template<typename Op>
inline elembinary::ElementWiseBinaryOpType GetOpType();

template<typename Op>
inline const char* GetOpTypeString();

template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::plus>() {
  return elembinary::kPlus;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::minus>() {
  return elembinary::kMinus;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::mul>() {
  return elembinary::kMul;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow::op::div>() {
  return elembinary::kDiv;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow_op::power>() {
  return elembinary::kPower;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow_op::maximum>() {
  return elembinary::kMaximum;
}
template<>
inline elembinary::ElementWiseBinaryOpType GetOpType<mshadow_op::minimum>() {
  return elembinary::kMinimum;
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

template<>
inline const char* GetOpTypeString<mshadow_op::power>() {
  return "_Power";
}

template<>
inline const char* GetOpTypeString<mshadow_op::maximum>() {
  return "_Maximum";
}

template<>
inline const char* GetOpTypeString<mshadow_op::minimum>() {
  return "_Minimum";
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
    Tensor<xpu, 2> lhs = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> rhs = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[elembinary::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[elembinary::kOut], F<ForwardOp>(lhs, rhs));
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
    Tensor<xpu, 2> m_out_grad = out_grad[elembinary::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> lhs_grad = in_grad[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> rhs_grad = in_grad[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
    switch (GetOpType<ForwardOp>()) {
    case elembinary::kPlus: {
      Assign(lhs_grad, req[elembinary::kLhs], F<mshadow_op::identity>(m_out_grad));
      Assign(rhs_grad, req[elembinary::kRhs], F<mshadow_op::identity>(m_out_grad));
      break;
    }
    case elembinary::kMinus: {
      Assign(lhs_grad, req[elembinary::kLhs], F<mshadow_op::identity>(m_out_grad));
      Assign(rhs_grad, req[elembinary::kRhs], F<mshadow_op::negation>(m_out_grad));
      break;
    }
    case elembinary::kMul: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      // rhs cannot do inplace
      CHECK_NE(req[elembinary::kRhs], kWriteInplace);
      Assign(rhs_grad, req[elembinary::kRhs], lhs_data * m_out_grad);
      Assign(lhs_grad, req[elembinary::kLhs], rhs_data * m_out_grad);
      break;
    }
    case elembinary::kDiv: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      // rhs cannot do inplace
      CHECK_NE(req[elembinary::kRhs], kWriteInplace);
      Assign(rhs_grad, req[elembinary::kRhs],
             F<mshadow_op::negation>(m_out_grad * lhs_data) / F<mshadow_op::square>(rhs_data));
      Assign(lhs_grad, req[elembinary::kLhs], m_out_grad / rhs_data);
      break;
    }
    case elembinary::kPower: {
      Tensor<xpu, 2> base_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> exponent_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> m_out_data = out_data[elembinary::kOut].FlatTo2D<xpu, real_t>(s);
      // rhs cannot do inplace
      CHECK_NE(req[elembinary::kRhs], kWriteInplace);
      Assign(rhs_grad, req[elembinary::kRhs],
             F<mshadow_op::log>(base_data) * m_out_data * m_out_grad);
      Assign(lhs_grad, req[elembinary::kLhs],
             exponent_data * F<mshadow_op::power>(base_data, exponent_data - 1) * m_out_grad);
      break;
    }
    case elembinary::kMaximum: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      Assign(lhs_grad, req[elembinary::kLhs],
             m_out_grad * F<mshadow_op::maximum_grad>(lhs_data, rhs_data));
      Assign(rhs_grad, req[elembinary::kRhs],
             m_out_grad * F<mshadow_op::minimum_grad>(lhs_data, rhs_data));
      break;
    }
    case elembinary::kMinimum: {
      Tensor<xpu, 2> lhs_data = in_data[elembinary::kLhs].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> rhs_data = in_data[elembinary::kRhs].FlatTo2D<xpu, real_t>(s);
      Assign(lhs_grad, req[elembinary::kLhs],
             m_out_grad * F<mshadow_op::minimum_grad>(lhs_data, rhs_data));
      Assign(rhs_grad, req[elembinary::kRhs],
             m_out_grad * F<mshadow_op::maximum_grad>(lhs_data, rhs_data));
      break;
    }
    }
  }
};  // class ElementWiseBinaryOp


template<typename xpu>
inline Operator* CreateElementWiseBinaryOp_(elembinary::ElementWiseBinaryOpType type) {
  switch (type) {
  case elembinary::kPlus:
    return new ElementWiseBinaryOp<xpu, mshadow::op::plus>();
  case elembinary::kMinus:
    return new ElementWiseBinaryOp<xpu, mshadow::op::minus>();
  case elembinary::kMul:
    return new ElementWiseBinaryOp<xpu, mshadow::op::mul>();
  case elembinary::kDiv:
    return new ElementWiseBinaryOp<xpu, mshadow::op::div>();
  case elembinary::kPower:
    return new ElementWiseBinaryOp<xpu, mshadow_op::power>();
  case elembinary::kMaximum:
    return new ElementWiseBinaryOp<xpu, mshadow_op::maximum>();
  case elembinary::kMinimum:
    return new ElementWiseBinaryOp<xpu, mshadow_op::minimum>();
  }
  LOG(FATAL) << "uknown op type";
  return NULL;
}

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateElementWiseBinaryOp(elembinary::ElementWiseBinaryOpType type);

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
    if (in_shape->at(elembinary::kLhs).ndim() != 0) {
      SHAPE_ASSIGN_CHECK(*in_shape, elembinary::kRhs, in_shape->at(elembinary::kLhs));
    } else if (in_shape->at(elembinary::kRhs).ndim() != 0) {
      in_shape->at(elembinary::kLhs) = in_shape->at(elembinary::kRhs);
    } else {
      return false;
    }
    const TShape &dshape = in_shape->at(elembinary::kLhs);
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
    case elembinary::kPlus:
    case elembinary::kMinus:
      return {out_grad[elembinary::kOut]};
    case elembinary::kMul:
    case elembinary::kDiv:
    case elembinary::kMaximum:
    case elembinary::kMinimum:
      return {out_grad[elembinary::kOut], in_data[elembinary::kLhs], in_data[elembinary::kRhs]};
    case elembinary::kPower:
      return {out_grad[elembinary::kOut], in_data[elembinary::kLhs], in_data[elembinary::kRhs],
              out_data[elembinary::kOut]};
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
    case elembinary::kPlus:
    case elembinary::kMinus:
    case elembinary::kMaximum:
    case elembinary::kMinimum:
      return {};
    case elembinary::kMul:
    case elembinary::kDiv:
    case elembinary::kPower:
      return {{out_grad[elembinary::kOut], in_grad[elembinary::kLhs]}};
    }
    LOG(FATAL) << "not reached";
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[elembinary::kLhs], out_data[elembinary::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
