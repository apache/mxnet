/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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

