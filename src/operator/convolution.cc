/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConvolutionParam param) {
  return new ConvolutionOp<cpu>(param);
}

Operator* ConvolutionProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ConvolutionParam);

MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

