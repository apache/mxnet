/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(DropoutParam param) {
  return new DropoutOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet


