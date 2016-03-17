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
  Operator *op = NULL;
  switch(param.dtype) {
  case mshadow::kFloat32:
    op = new ConvolutionOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
  	op = new ConvolutionOp<cpu, double>(param);
  	break;
  case mshadow::kFloat16:
  	LOG(FATAL) << "float16 is currently only supported by CuDNN version.";
  	break;
  default:
  	LOG(FATAL) << "Unsupported type " << param.dtype;
  }
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ConvolutionProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
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

