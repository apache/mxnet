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
StaticOperator* FullyConnectSymbol::Bind_<cpu>(Context ctx) const {
  return new FullyConnectOp<cpu>(param_);
}

// put this after the template specialization
StaticOperator* FullyConnectSymbol::Bind(Context ctx) const {
  if (ctx.dev_mask == cpu::kDevMask) {
    return Bind_<cpu>(ctx);
  } else {
    #if MXNET_USE_CUDA
    return Bind_<gpu>(ctx);
    #else
    LOG(FATAL) << "GPU is not enabled";
    return NULL;
    #endif
  }
}

// register the symbol
REGISTER_ATOMIC_SYMBOL(FullyConnected, FullyConnectSymbol);

}  // namespace op
}  // namespace mxnet
