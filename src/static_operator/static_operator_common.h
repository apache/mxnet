/*!
 * Copyright (c) 2015 by Contributors
 * \file static_operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 *   common type definitions
 * \author Bing Xu
*/
#ifndef MXNET_STATIC_OPERATOR_STATIC_OPERATOR_COMMON_H_
#define MXNET_STATIC_OPERATOR_STATIC_OPERATOR_COMMON_H_

#include <dmlc/logging.h>
#include <mxnet/static_operator.h>
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
                   GradReqType req,
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

/*! \brief type of operators */
enum OpType {
  kReLU = 0,
  kFullc = 1,
  kConv = 2,
  kMaxPooling = 3,
  kAvgPooling = 4,
  kSumPooling = 5,
  kFlatten = 6,
  kReshape = 7,
  kDropout = 8,
};

/*!
 * \brief device invariant function to create operators
 * \param type the type of operator
 */
template<typename xpu>
StaticOperator *CreateOperator(OpType type);
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_STATIC_OPERATOR_STATIC_OPERATOR_COMMON_H_
