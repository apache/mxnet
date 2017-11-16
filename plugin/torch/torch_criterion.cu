/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
#include "./torch_criterion-inl.h"
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(TorchCriterionParam param) {
  return new TorchCriterionOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
