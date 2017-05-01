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
.add_argument("data", "NDArray-or-Symbol", "Input data to the DeconvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(DeconvolutionParam::__FIELDS__())
.describe("Applies deconvolution to input and adds a bias.");

}  // namespace op
}  // namespace mxnet
