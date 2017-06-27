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
Operator* CreateOp<cpu>(DeconvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DeconvolutionOp<cpu, DType>(param);
  });
  return op;
}

Operator* DeconvolutionProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0), in_shape, &out_shape, ctx);
}

DMLC_REGISTER_PARAMETER(DeconvolutionParam);

MXNET_REGISTER_OP_PROPERTY(Deconvolution, DeconvolutionProp)
.add_argument("data", "NDArray-or-Symbol", "Input tensor to the deconvolution operation.")
.add_argument("weight", "NDArray-or-Symbol", "Weights representing the kernel.")
.add_argument("bias", "NDArray-or-Symbol", "Bias added to the result after the deconvolution "
    "operation.")
.add_arguments(DeconvolutionParam::__FIELDS__())
.describe("Computes 2D transposed convolution (aka fractionally strided convolution) of the "
    "input tensor. This operation can be seen as the gradient of Convolution operation with "
    "respect to its input. Convolution usually reduces the size of the input. Transposed "
    "convolution works the other way, going from a smaller input to a larger output while "
    "preserving the connectivity pattern.");

}  // namespace op
}  // namespace mxnet
