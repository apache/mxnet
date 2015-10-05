/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function-inl.h
 * \brief The real implementation of NDArray functions.
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_

#include <vector>
#include "./ndarray_function.h"
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
#define DECL_SCALAR(XPU, OP, FUN, REVERSE)                              \
  template<>                                                            \
  void Eval<XPU, OP, REVERSE>(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP, REVERSE>(lhs, rhs, ret, ctx);                          \
  }
#endif

#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace ndarray {
// true implementation
template<typename xpu, typename OP>
inline void EvalBinary_(const TBlob &lhs, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, real_t>(s),
                                   rhs.FlatTo2D<xpu, real_t>(s));
}

template<typename xpu, typename OP>
inline void EvalDot_(const TBlob &lhs, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->FlatTo2D<xpu, real_t>(s)
    = dot(lhs.FlatTo2D<xpu, real_t>(s),
          rhs.FlatTo2D<xpu, real_t>(s));
}

template<typename xpu, typename OP, bool reverse>
inline void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (reverse) {
    ret->FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(rhs, lhs.FlatTo2D<xpu, real_t>(s));
  } else {
    ret->FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, real_t>(s), rhs);
  }
}


template<>
void EvalClip<DEVICE>(const TBlob &src, const real_t &a_min, const real_t &a_max,
                      TBlob *ret, RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->FlatTo2D<xpu, real_t>(s)
    = F<ClipMax::mshadow_op>(
        F<ClipMin::mshadow_op>(src.FlatTo2D<xpu, real_t>(s), a_min),
        a_max);
}

template<>
void EvalRandom<DEVICE, UniformDistribution>(
    const real_t &a,
    const real_t &b,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2, real_t> tmp = ret->FlatTo2D<xpu, real_t>(s);
  mshadow::Random<xpu> *prnd = resource.get_random<xpu>(s);
  prnd->SampleUniform(&tmp, a, b);
}

template<>
void EvalRandom<DEVICE, GaussianDistribution>(
    const real_t &mu,
    const real_t &sigma,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2, real_t> tmp = ret->FlatTo2D<xpu, real_t>(s);
  mshadow::Random<xpu> *prnd = resource.get_random<xpu>(s);
  prnd->SampleGaussian(&tmp, mu, sigma);
}

template<>
void Eval<DEVICE>(const real_t &rhs, TBlob *ret, RunContext ctx) {
  mshadow::Stream<DEVICE> *s = ctx.get_stream<DEVICE>();
  ret->FlatTo2D<DEVICE, real_t>(s) = rhs;
}

template<>
void ElementwiseSum<DEVICE>(const std::vector<TBlob> source,
                            TBlob *dst,
                            RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> out = dst->FlatTo2D<xpu, real_t>(s);

  switch (source.size()) {
    case 2: {
      Tensor<xpu, 2> in_0 = source[0].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> in_1 = source[1].FlatTo2D<xpu, real_t>(s);
      out = in_0 + in_1;
      break;
    }
    case 3: {
      Tensor<xpu, 2> in_0 = source[0].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> in_1 = source[1].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> in_2 = source[2].FlatTo2D<xpu, real_t>(s);
      out = in_0 + in_1 + in_2;
      break;
    }
    case 4: {
      Tensor<xpu, 2> in_0 = source[0].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> in_1 = source[1].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> in_2 = source[2].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> in_3 = source[3].FlatTo2D<xpu, real_t>(s);
      out = in_0 + in_1 + in_2 + in_3;
      break;
    }
    default: {
      Tensor<xpu, 2> in_0 = source[0].FlatTo2D<xpu, real_t>(s);
      out = F<op::identity>(in_0);
      for (size_t i = 1; i < source.size(); ++i) {
        out += source[i].FlatTo2D<xpu, real_t>(s);
      }
      break;
    }
  }
}

// declarations
DECL_BINARY(DEVICE, Dot, EvalDot_)
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
}  // namespace ndarray
}  // namespace mxnet

#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
