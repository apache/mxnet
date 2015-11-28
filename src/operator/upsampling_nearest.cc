/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling_nearest.cc
 * \brief
 * \author Bing Xu
*/


#include "./upsampling_nearest-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(UpSamplingNearestParam param) {
  return new UpSamplingNearestOp<cpu>(param);
}

Operator* UpSamplingNearestProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(UpSamplingNearestParam);

MXNET_REGISTER_OP_PROPERTY(UpSamplingNearest, UpSamplingNearestProp)
.describe("Perform simple nearest neighboor up sampling to inputs")
.add_argument("data", "Symbol", "Input data to the up sampling operator.")
.add_arguments(UpSamplingNearestParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
