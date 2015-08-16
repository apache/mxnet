/*!
 * Copyright (c) 2015 by Contributors
 * \file  operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_OPERATOR_COMMON_H_
#define MXNET_OPERATOR_OPERATOR_COMMON_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>

namespace mxnet {
namespace op {
/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
template<typename OType, typename Exp>
inline void Assign(OType &out, // NOLINT(*)
                   OpReqType req,
                   const Exp &exp) {
  switch (req) {
    case kNullOp: break;
    case kWriteTo:
    case kWriteInplace: out = exp; break;
    case kAddTo: out += exp; break;
    default: LOG(FATAL) << "not reached";
  }
}
/*!
 * \brief assign shape to out if out is unknown
 *  otherwise check consistency
 * \param out the output shape to be stored
 * \param shape the infered shape
 */
template<typename TS>
inline void ShapeAssignCheck(TShape &out, const TS &shape) { // NOLINT(*)
  if (out.ndim() == 0) {
    out = shape;
  } else {
    CHECK(out == shape) << "InferShape:: shape inconsistent";
  }
}

// helper macro to implement bind dispatch
#if MXNET_USE_CUDA
#define DO_BIND_DISPATCH(Method, ...)                                \
    if (ctx.dev_mask == cpu::kDevMask) {                             \
      return Method<cpu>(__VA_ARGS__);                               \
    } else {                                                         \
      return Method<gpu>(__VA_ARGS__);                               \
    }
#else
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask == cpu::kDevMask) {                               \
    return Method<cpu>(__VA_ARGS__);                                 \
  } else {                                                           \
    LOG(FATAL) << "GPU is not enabled";                              \
    return nullptr;                                                  \
  }
#endif

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_OPERATOR_COMMON_H_
