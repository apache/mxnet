/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_loss.cc
 * \brief caffe loss 
 * \author Haoran Wang 
*/
#include "./caffe_loss-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(CaffeLossParam param) {
  return new CaffeLoss<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator *CaffeLossProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CaffeLossParam);

MXNET_REGISTER_OP_PROPERTY(CaffeLoss, CaffeLossProp)
.describe("Caffe loss layer")
.add_arguments(CaffeLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
