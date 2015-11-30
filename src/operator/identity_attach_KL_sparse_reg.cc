/*!
 * Copyright (c) 2015 by Contributors
 * \file identity_attach_KL_sparse_reg.cc
 * \brief\
*/
#include "./identity_attach_KL_sparse_reg-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(IdentityAttachKLSparseRegParam param) {
  return new IdentityAttachKLSparseRegOp<cpu>(param);
}

Operator *IdentityAttachKLSparseRegProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(IdentityAttachKLSparseRegParam);

MXNET_REGISTER_OP_PROPERTY(IdentityAttachKLSparseReg, IdentityAttachKLSparseRegProp)
.describe("Apply a sparse regularization to the output a sigmoid activation function.")
.add_argument("data", "Symbol", "Input data.")
.add_arguments(IdentityAttachKLSparseRegParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet

