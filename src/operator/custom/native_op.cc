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
 * \file native_op.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./native_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(NativeOpParam param) {
  return new NativeOp<cpu>(param);
}

Operator* NativeOpProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(NativeOpParam);

MXNET_REGISTER_OP_PROPERTY(_Native, NativeOpProp)
.describe("Stub for implementing an operator implemented in native frontend language.")
.add_argument("data", "NDArray-or-Symbol[]", "Input data for the custom operator.")
.add_arguments(NativeOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
