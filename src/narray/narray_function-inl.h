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
#define DECL_SCALAR(XPU, OP, FUN)                                       \
  template<>                                                            \
  void Eval<XPU, OP>(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                   \
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

template<typename xpu, typename OP>
inline void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = static_cast<mshadow::Stream<xpu>*>(ctx.stream);
  ret->FlatTo2D<xpu, real_t>(s)
    = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, real_t>(s), rhs);
}
// declarations
DECL_BINARY(DEVICE, Plus, EvalBinary_)
DECL_BINARY(DEVICE, Minus, EvalBinary_)
DECL_BINARY(DEVICE, Mul, EvalBinary_)
DECL_BINARY(DEVICE, Div, EvalBinary_)
DECL_SCALAR(DEVICE, Plus, EvalScalar_)
DECL_SCALAR(DEVICE, Minus, EvalScalar_)
DECL_SCALAR(DEVICE, Mul, EvalScalar_)
DECL_SCALAR(DEVICE, Div, EvalScalar_)
}  // namespace narray
}  // namespace mxnet

#endif  // MXNET_NARRAY_NARRAY_FUNCTION_INL_H_
