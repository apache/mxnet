/*!
 * Copyright (c) 2015 by Contributors
 * \file prod_sum.cc
 * \brief product sum op
 * \author Junyuan Xie
*/
#include "./prod_sum-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ProdSumParam param) {
  return new ProdSumOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ProdSumProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ProdSumParam);

MXNET_REGISTER_OP_PROPERTY(ProdSum, ProdSumProp)
.describe("Compute dot product along one dim of 2 tensors.")
.add_arguments(ProdSumParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

