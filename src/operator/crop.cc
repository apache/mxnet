/*!
 * Copyright (c) 2015 by Contributors
 * \file crop.cc
 * \brief
 * \author Wei Wu
*/

#include "./crop-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(CropParam param) {
  return new CropOp<cpu>(param);
}

Operator* CropProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CropParam);

MXNET_REGISTER_OP_PROPERTY(Crop, CropProp)
.describe("Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or "
"with width and height of the second input symbol, i.e., with one input, we need h_w to "
"specify the crop height and width, otherwise the second input symbol's size will be used")
.add_argument("data", "Symbol or Symbol[]", "Tensor or List of Tensors, the second input "
"will be used as crop_like shape reference")
.add_arguments(CropParam::__FIELDS__())
.set_key_var_num_args("num_args");
}  // namespace op
}  // namespace mxnet
