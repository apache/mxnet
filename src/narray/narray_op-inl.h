#include "./narray_op.h"
// this file will be included twice by CPU and GPU
// macro to help specialize evaluation function
#ifndef DECL_BINARY
#define DECL_BINARY(XPU, OP, FUN)                                         \
  template<>                                                            \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &rhs, TBlob ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                    \
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
inline void Eval_(const TBlob &lhs, const TBlob &rhs, TBlob ret, RunContext ctx) {  
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = static_cast<mshadow::Stream<xpu>*>(ctx.stream);
  ret.FlatTo2D<xpu, real_t>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, real_t>(s),
                                   rhs.FlatTo2D<xpu, real_t>(s));
}
// declarations
DECL_BINARY(DEVICE, Plus, Eval_)
DECL_BINARY(DEVICE, Minus, Eval_)
DECL_BINARY(DEVICE, Mul, Eval_)
DECL_BINARY(DEVICE, Div, Eval_)
}  // namespace narray
}  // namespace mxnet
