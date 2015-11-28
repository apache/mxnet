/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling.cc
 * \brief
 * \author Bing Xu
*/


#include "./upsampling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(UpSamplingParam param) {
  return new UpSamplingOp<cpu>(param);
}

Operator* UpSamplingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(UpSamplingParam);

MXNET_REGISTER_OP_PROPERTY(UpSampling, UpSamplingProp)
.describe("Perform simple nearest neighboor up sampling to inputs")
.add_argument("data", "Symbol", "Input data to the up sampling operator.")
.add_arguments(UpSamplingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
