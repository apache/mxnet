/*!
 * Copyright (c) 2015 by Contributors
 * \file make_loss.cu
 * \brief special layer for propagating loss
*/
#include "./make_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MakeLossParam param) {
  return new MakeLossOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

