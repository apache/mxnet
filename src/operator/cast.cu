/*!
 * Copyright (c) 2015 by Contributors
 * \file cast.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./cast-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(CastParam param) {
  return new CastOp<gpu>();
}
}  // op
}  // namespace mxnet

