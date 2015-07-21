  /*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_sym.cu
 * \brief fully connect operator symbol
*/
#include "../static_operator/fully_connect_op-inl.h"

namespace mxnet {
namespace op {
  template<>
  StaticOperator* FullyConnectSymbol::Bind_<gpu>(Context ctx) const {
    return new FullyConnectOp<gpu>(param_);
  }
} // namespace op
} // namespace mxnet