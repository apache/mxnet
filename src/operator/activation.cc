/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief
 * \author Bing Xu
*/

#include <mxnet/registry.h>
#include "./activation-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateActivationOp<cpu>(ActivationOpType type) {
  switch (type) {
    case kReLU: return new ActivationOp<cpu, act::relu, act::relu_grad>();
    default: return NULL;
  }
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ActivationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateActivationOp, type_);
}

REGISTER_OP_PROPERTY(Activation, ActivationProp);
}  // namespace op
}  // namespace mxnet

