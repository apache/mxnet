/*!
 * Copyright (c) 2015 by Contributors
 * \file identity_attach_KL_sparse_reg.cc
 * \brief\
*/
#include "./identity_attach_KL_sparse_reg-inl.h"
#include <nnvm/op_attr_types.h>

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
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_arguments(IdentityAttachKLSparseRegParam::__FIELDS__());

NNVM_REGISTER_OP(IdentityAttachKLSparseReg)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 1) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      }
    });
}  // namespace op
}  // namespace mxnet

