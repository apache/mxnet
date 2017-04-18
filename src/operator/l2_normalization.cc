/*!
 * Copyright (c) 2015 by Contributors
 * \file l2_normalization.cc
 * \brief l2 normalization operator
*/
#include "./l2_normalization-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(L2NormalizationParam param) {
  return new L2NormalizationOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* L2NormalizationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(L2NormalizationParam);

MXNET_REGISTER_OP_PROPERTY(L2Normalization, L2NormalizationProp)
.describe(R"code(Normalize the input instance using a L2 norm.

For 1-D NDArray, it computes::

  out = data / sqrt(sum(data ** 2) + eps)

The parameter ``mode`` can specify the dimension along which to compute L2 norm.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to normalize.")
.add_arguments(L2NormalizationParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
