/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./volumetric_concat-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(VolumetricConcatParam param) {
  return new VolumetricConcatOp<cpu>(param);
}

Operator* VolumetricConcatProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(VolumetricConcatParam);

MXNET_REGISTER_OP_PROPERTY(VolumetricConcat, VolumetricConcatProp)
.describe("Perform an feature concat on channel dim (dim 1) over all the inputs.")
.add_arguments(VolumetricConcatParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet

