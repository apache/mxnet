  /*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_sym.cu
 * \brief fully connect operator symbol
*/
#include "./fully_connect_sym-inl.h"

namespace mxnet {
  template <>
  StaticOperator* FullyConnectSymbol::Bind<gpu>(Context ctx) const {
    return Bind_<gpu>(ctx);
  }
}
