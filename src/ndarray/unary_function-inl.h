/*!
 *  Copyright (c) 2015 by Contributors
 * \file unary-function-inl.h
 * \brief the real execution functions of ndarray operations
 */
#ifndef MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
#define MXNET_NDARRAY_UNARY_FUNCTION_INL_H_

#include "../common/tblob_op_registry.h"
#include "../operator/mshadow_op.h"

#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace ndarray {

template<typename xpu, typename OP>
void EvalUnary_(const TBlob &src,
                TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->FlatTo2D<xpu, real_t>(s)
      = F<OP>(src.FlatTo2D<xpu, real_t>(s));
}

// helper macro to register mshadow element-wise unary opts
// usually you only need to use this to register common operations
#define REGISTER_MSHADOW_UNARY(Name, Op)            \
  MXNET_REGISTER_TBLOB_FUN(Name, DEVICE)            \
  .set_function(DEVICE::kDevMask, EvalUnary_<DEVICE, Op>)


// register all unary operations here
REGISTER_MSHADOW_UNARY(square, op::mshadow_op::square)
.describe("Take square of the src");

REGISTER_MSHADOW_UNARY(sqrt, op::mshadow_op::square_root)
.describe("Take square root of the src");

}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
