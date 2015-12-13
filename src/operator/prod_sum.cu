/*!
 * Copyright (c) 2015 by Contributors
 * \file prod_sum.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./prod_sum-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ProdSumParam param) {
  return new ProdSumOp<gpu>(param);
}
}  // op
}  // namespace mxnet
