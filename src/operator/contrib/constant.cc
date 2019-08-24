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
 * Copyright (c) 2018 by Contributors
 * \file constant.cc
*/

#include "./constant-inl.h"
#include "../tensor/elemwise_binary_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

inline bool ConstantType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const ConstantParam& param_ = nnvm::get<ConstantParam>(attrs.parsed);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param_.dtype);
  return true;
}



DMLC_REGISTER_PARAMETER(ConstantParam);
NNVM_REGISTER_OP(_contrib_constant)
.describe(R"code(Creates a constant tensor for a value.
Example::
  v1 = (1, 2)
  constant_op = symbol.contrib.constant(value=v1)
  executor = constant_op.simple_bind(ctx=cpu())
  executor.forward(is_train=True)
  executor.outputs
  [ -1.  2.]
)code" ADD_FILELINE)
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConstantParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ConstantShape)
.set_attr<nnvm::FInferType>("FInferType", ConstantType)
.set_attr<FCompute>("FCompute<cpu>", ConstantForward<cpu, ConstantParam>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(ConstantParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet