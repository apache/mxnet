/*!
 * Copyright (c) 2015 by Contributors
 * \file assign_helper.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_HELPER_H_
#define MXNET_OPERATOR_HELPER_H_
#include "activation_op-inl.h"
#include "mshadow_op.h"

namespace mxnet {
namespace op {

enum OpType {
  kReLU = 0,
};


template<typename xpu, typename Exp>
inline void Assign(const Exp &exp,
            const mshadow::Tensor<xpu,2> &out,
            const Operator::GradReqType &req) {
  switch (req) {
    case Operator::kNullOp:
      break;
    case Operator::kWriteTo:
    case Operator::kWriteInplace:
      break;
    case Operator::kAddTo:
      break;
  }
}

template<typename xpu>
Operator *OperatorFactory(OpType type) {
  switch (type) {
    case kReLU:
      return new ActivationOp<xpu, relu, relu_grad>();

  };
  return NULL;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_HELPER_H_
