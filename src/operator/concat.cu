/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cu
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConcatParam param) {
  return new ConcatOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

