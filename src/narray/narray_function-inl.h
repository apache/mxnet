/*!
 *  Copyright (c) 2015 by Contributors
 * \file narray_function-inl.h
 * \brief
 */
#ifndef MXNET_NARRAY_NARRAY_FUNCTION_INL_H_
#define MXNET_NARRAY_NARRAY_FUNCTION_INL_H_
#include "./narray_function.h"
// this file will be included twice by CPU and GPU
// macro to help specialize evaluation function
#ifndef DECL_BINARY
#define DECL_BINARY(XPU, OP, FUN)                                       \
  template<>                                                            \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                   \
  }
#endif

#ifndef DECL_SCALAR
#define DECL_SCALAR(XPU, OP, FUN, REVERSE)                                       \
  template<>                                                            \
  void Eval<XPU, OP, REVERSE>(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP, REVERSE>(lhs, rhs, ret, ctx);                                   \
  }
#endif

#ifndef DECL_SETVALUE
#define DECL_SETVALUE(XPU)                                       \
  template<>                                                            \
  void Eval<XPU>(const real_t &rhs, TBlob *ret, RunContext ctx) { \
    mshadow::Stream<XPU> *s = static_cast<mshadow::Stream<XPU>*>(ctx.stream);    \
    ret->FlatTo2D<XPU, real_t>(s) = rhs;                          \
  }
#endif


#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace narray {
// true implementation
template<typename xpu, typename OP>
inline void EvalBinary_(const TBlob &lhs, const TBlob &rhs,
                  TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = static_cast<mshadow::Stream<xpu>*>(ctx.stream);
  ret->FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, real_t>(s),
                                   rhs.FlatTo2D<xpu, real_t>(s));
}

template<typename xpu, typename OP, bool reverse>
inline void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = static_cast<mshadow::Stream<xpu>*>(ctx.stream);
  if (reverse) {
    ret->FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(rhs, lhs.FlatTo2D<xpu, real_t>(s));
  } else {
    ret->FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, real_t>(s), rhs);
  }
}


// declarations
DECL_BINARY(DEVICE, Plus, EvalBinary_)
DECL_BINARY(DEVICE, Minus, EvalBinary_)
DECL_BINARY(DEVICE, Mul, EvalBinary_)
DECL_BINARY(DEVICE, Div, EvalBinary_)
DECL_SCALAR(DEVICE, Plus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, true)
DECL_SCALAR(DEVICE, Div, EvalScalar_, true)
// for reverse seq
DECL_SCALAR(DEVICE, Plus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, false)
DECL_SCALAR(DEVICE, Div, EvalScalar_, false)
//
DECL_SETVALUE(DEVICE)
}  // namespace narray
}  // namespace mxnet

#endif  // MXNET_NARRAY_NARRAY_FUNCTION_INL_H_
