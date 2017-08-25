/*!
 * Copyright (c) 2015 by Contributors
 * \file l2_normalization.cu
 * \brief l2 normalization operator
*/
#include "./l2_normalization-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(L2NormalizationParam param) {
  return new L2NormalizationOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
