/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Junyuan Xie
*/
#include "./torch_criterion-inl.h"
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(TorchCriterionParam param) {
  return new TorchCriterionOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *TorchCriterionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(TorchCriterionParam);

MXNET_REGISTER_OP_PROPERTY(TorchCriterion, TorchCriterionProp)
.describe("Criterions from torch.")
.add_arguments(TorchCriterionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
