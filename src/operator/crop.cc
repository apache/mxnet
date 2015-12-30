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
.describe("Crop the 2th and 3th dim of input data, with the corresponding size of crop_like.")
// .add_argument("data", "Symbol", "Input data to the CropOp.")
// .add_argument("crop_like", "Symbol", "crop_like data to the CropOp.")
.add_arguments(CropParam::__FIELDS__())
.set_key_var_num_args("num_args");
}  // namespace op
}  // namespace mxnet
