/*!
 * Copyright (c) 2015 by Contributors
 * \file sparse_reg.cc
 * \brief\
*/
#include "./sparse_reg-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SparseRegParam param) {
  return new SparseRegOp<cpu>(param);
}

Operator *SparseRegProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SparseRegParam);

MXNET_REGISTER_OP_PROPERTY(SparseReg, SparseRegProp)
.describe("Apply a sparse regularization to the output a sigmoid activation function.")
.add_argument("data", "Symbol", "Input data.")
.add_arguments(SparseRegParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet

