/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
#include "./torch_module-inl.h"
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(TorchModuleParam param, TorchState* torchState) {
  return new TorchModuleOp<cpu>(param, torchState);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *TorchModuleProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_, torchState_);
}

DMLC_REGISTER_PARAMETER(TorchModuleParam);

MXNET_REGISTER_OP_PROPERTY(TorchModule, TorchModuleProp)
.describe("Modules from torch.")
.add_arguments(TorchModuleParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
