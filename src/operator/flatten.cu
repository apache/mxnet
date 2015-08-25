/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./flatten-inl.h"


namespace mxnet {
namespace op {
template<>
  Operator *CreateOp<gpu>() {
  return new FlattenOp<gpu>();
}

}  // namespace op
}  // namespace mxnet
