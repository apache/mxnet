/*!
 * Copyright (c) 2015 by Contributors
 * \file assign_helper.h
 * \brief assign gradient
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ASSIGN_HELPER_H_
#define MXNET_OPERATOR_ASSIGN_HELPER_H_
namespace mxnet {
namespace op {
template<typename xpu, typename Exp, int dim, typename DType>
inline void Assign(mshadow::Tensor<xpu, dim, DType> &out,
                   const Exp &exp,
                   const Operator::GradReqType &req) {
  switch (req) {
    case Operator::kNullOp:
      break;
    case Operator::kWriteTo:
    case Operator::kWriteInplace:
      out = exp;
      break;
    case Operator::kAddTo:
      out += exp;
      break;
  }
}
}  //namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ASSIGN_HELPER_H_
