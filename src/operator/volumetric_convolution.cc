/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./volumetric_convolution-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(VolumetricConvolutionParam param) {
  return new VolumetricConvolutionOp<cpu>(param);
}

Operator* VolumetricConvolutionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(VolumetricConvolutionParam);

MXNET_REGISTER_OP_PROPERTY(VolumetricConvolution, VolumetricConvolutionProp)
.add_argument("data", "Symbol", "Input data to the VolumetricConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(VolumetricConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

