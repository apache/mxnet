/*!
 * Copyright (c) 2015 by Contributors
 * \file deconvolution.cc
 * \brief
 * \author Wei Wu
*/

#include "./deconvolution-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(DeconvolutionParam param) {
  return new DeconvolutionOp<cpu>(param);
}

Operator* DeconvolutionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(DeconvolutionParam);

MXNET_REGISTER_OP_PROPERTY(Deconvolution, DeconvolutionProp)
.add_argument("data", "Symbol", "Input data to the DeconvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(DeconvolutionParam::__FIELDS__())
.describe("Apply deconvolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
