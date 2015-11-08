/*!
 * Copyright (c) 2015 by Contributors
 * \file Scalar.cc
 * \brief Scalar op
 * \author Bing Xu
*/
#include "./scalar-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ScalarParam param) {
    return new ScalarOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ScalarProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ScalarParam);

MXNET_REGISTER_OP_PROPERTY(Scalar, ScalarProp)
.describe("Apply Scalar function to input.")
.add_argument("value", "float", "Input data to Scalar function.")
.add_arguments(ScalarParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

