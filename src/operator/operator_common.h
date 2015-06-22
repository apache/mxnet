/*!
 * Copyright (c) 2015 by Contributors
 * \file operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 *   common type definitions
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_OPERATOR_COMMON_H_
#define MXNET_OPERATOR_OPERATOR_COMMON_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>

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
inline void Assign(OType &out,
                   Operator::GradReqType req,
                   const Exp &exp) {
  switch (req) {
    case Operator::kNullOp: break;
    case Operator::kWriteTo:
    case Operator::kWriteInplace: out = exp; break;
    case Operator::kAddTo: out += exp; break;
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
inline void ShapeAssignCheck(TShape &out, const TS &shape) {
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
};

/*!
 * \brief device invariant function to create operators
 * \param type the type of operator
 * \tparam xpu the device type we are at
 */
template<typename xpu>
Operator *CreateOperator(OpType type);
}  //namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_COMMON_H_
