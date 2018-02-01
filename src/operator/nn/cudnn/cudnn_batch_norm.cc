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
 * \file cudnn_batch_norm.cc
 * \brief
 * \author Junyuan Xie
*/

#include "./cudnn_batch_norm-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {
#if CUDNN_MAJOR >= 4
template<>
Operator *CreateOp_CuDNNv4<cpu>(BatchNormParam param) {
  LOG(FATAL) << "CuDNNBatchNormOp is only available for gpu.";
  return NULL;
}

Operator *CuDNNBatchNormProp::CreateOperator(Context ctx) const {
#if CUDNN_MAJOR >= 5
  LOG(FATAL) << "CuDNNBatchNorm is merged into BatchNorm for cudnn version above v5."
                "Use the later instead.";
  return nullptr;
#else
  DO_BIND_DISPATCH(CreateOp_CuDNNv4, param_);
#endif
}

MXNET_REGISTER_OP_PROPERTY(CuDNNBatchNorm, CuDNNBatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_arguments(BatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(CuDNNBatchNorm)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 3) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      } else if (index == 4) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      }
    });
#endif  // CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
