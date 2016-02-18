/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
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
.describe("Crop the 2nd and 3rd dim of input data, with the corresponding size of w_h or "
"with width and height of the second input symbol")
.add_arguments(CropParam::__FIELDS__())
.set_key_var_num_args("num_args");
}  // namespace op
}  // namespace mxnet
