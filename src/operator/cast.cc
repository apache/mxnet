/*!
 * Copyright (c) 2015 by Contributors
 * \file cast.cc
 * \brief cast op
 * \author Junyuan Xie
*/
#include "./cast-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CastParam param) {
  return new CastOp<cpu>();
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *CastProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CastParam);

MXNET_REGISTER_OP_PROPERTY(Cast, CastProp)
.describe("Cast array to a different data type.")
.add_argument("data", "Symbol", "Input data to cast function.")
.add_arguments(CastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

