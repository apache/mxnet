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
 * \file np_constraint_check.cc
 * \brief helper function for constraint check
 */

#include "./np_constraint_check.h"

namespace mxnet {
namespace op {

template<>
void GetReduceOutput<cpu>(mshadow::Stream<cpu> *s, const TBlob &output_blob, bool *red_output) {
  *red_output = static_cast<bool>(*output_blob.dptr<bool>());
}

inline bool ConstraintCheckShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  // Only 1-D support is supported.
  // CHECK_EQ(in_attrs->at(0).ndim(), 1U) << "Only 1-D input is supported.";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
  return true;
}

inline bool ConstraintCheckType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_attrs,
                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK(in_attrs->at(0) == mshadow::kBool);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(ConstraintCheckParam);

NNVM_REGISTER_OP(_npx_constraint_check)
.describe(R"code(Check if all the elements in a 1-D boolean array is true.
If not, exception will be raised with given error message.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ConstraintCheckParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"input"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ConstraintCheckShape)
.set_attr<nnvm::FInferType>("FInferType", ConstraintCheckType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", ConstraintCheckForward<cpu>)
.add_argument("input", "NDArray-or-Symbol", "Input boolean array")
.add_arguments(ConstraintCheckParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
