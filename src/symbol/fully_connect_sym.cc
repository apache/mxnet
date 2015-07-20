  /*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_sym.cc
 * \brief fully connect operator symbol
*/
#include "./fully_connect_sym-inl.h"

namespace mxnet {
  template <>
  StaticOperator* FullyConnectSymbol<cpu>::Bind(Context ctx) const {
    return Bind_(ctx);
  }
}
