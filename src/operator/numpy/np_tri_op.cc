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
* Copyright (c) 2019 by Contributors
* \file np_tri_op.cc
* \brief CPU implementation of numpy tri operator
*/

#include "./np_tri_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(TriParam);

inline bool TriOpShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector* in_attrs,
                      mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);

  const TriParam& param = nnvm::get<TriParam>(attrs.parsed);
  nnvm::dim_t M = param.M.has_value() ? param.M.value() : param.N;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(param.N, M));

  return shape_is_known(out_attrs->at(0));
}

inline bool TriOpType(const nnvm::NodeAttrs& attrs,
                      std::vector<int> *in_attrs,
                      std::vector<int> *out_attrs) {
  CHECK_GE(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);

  const TriParam& param = nnvm::get<TriParam>(attrs.parsed);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);

  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_tri)
.set_attr_parser(ParamParser<TriParam>)
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", TriOpShape)
.set_attr<nnvm::FInferType>("FInferType", TriOpType)
.set_attr<FCompute>("FCompute<cpu>", TriOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(TriParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
