/*!
 * Copyright (c) 2015 by Contributors
 * \file assign_helper.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_HELPER_H_
#define MXNET_OPERATOR_HELPER_H_
#include "activation_op-inl.h"
#include "fully_connect_op-inl.h"
#include "mshadow_op.h"

namespace mxnet {
namespace op {

enum OpType {
  kReLU = 0,
  kFullc = 1,
};

template<typename xpu>
Operator *OperatorFactory(OpType type) {
  switch (type) {
    case kReLU:
      return new ActivationOp<xpu, relu, relu_grad>();
    case kFullc:
      return new FullyConnectOp<xpu>();

  };
  return NULL;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_HELPER_H_
