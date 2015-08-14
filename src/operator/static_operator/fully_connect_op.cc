/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_sym.cc
 * \brief fully connect operator symbol
*/
#include <mxnet/registry.h>
#include "../static_operator/fully_connect_op-inl.h"
namespace mxnet {
namespace op {
template<>
StaticOperator* CreateFullyConnectedOp<cpu>(Param param) {
  return new FullyConnectOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
StaticOperator* FullyConnectSymbol::Bind(Context ctx) const {
  DO_BIND_DISPATCH(CreateFullyConnectedOp, param_);
}

REGISTER_ATOMIC_SYMBOL(FullyConnected, FullyConnectSymbol);
}  // namespace op
}  // namespace mxnet
