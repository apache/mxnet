/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_sym.cu
 * \brief fully connect operator symbol
*/
#include "./fully_connect_op-inl.h"
namespace mxnet {
namespace op {

template<>
StaticOperator* CreateFullyConnectedOp<gpu>(Param param) {
  return new FullyConnectOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
