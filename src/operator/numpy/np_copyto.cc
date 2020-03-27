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
 * \file np_copyto.cc
 * \brief CPU Implementation of numpy-compatible copyto
 */

#include "./np_copyto-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyCopytoParam);

bool NumpyCopytoType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_type,
                     std::vector<int> *out_type) {
  const NumpyCopytoParam& param = nnvm::get<NumpyCopytoParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), 0U);
  if (!param.where.has_value())
    CHECK_EQ(in_type->at(2), mshadow::kBool);
  if (param.casting == copyto_casting_enum::no) {
    CHECK_EQ(in_type->at(0), in_type->at(1))
    << "Cannot cast scalar from dtype(" << in_type->at(0) << ") to dtype("
    << in_type->at(1) << ") according to the rule 'no'\n";
  }
  return true;
}

NNVM_REGISTER_OP(_npi_copy2)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const NumpyCopytoParam& param = nnvm::get<NumpyCopytoParam>(attrs.parsed);
    int num_inputs = 3;
    if (param.src.has_value()) {
      num_inputs -= 1;
    }
    if (param.where.has_value()) {
      num_inputs -= 1;
    }
    return num_inputs;
  })
.set_num_outputs(0)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyCopytoParam& param = nnvm::get<NumpyCopytoParam>(attrs.parsed);
    int num_inputs = 1;
    if (!param.src.has_value()) {
      num_inputs += 1;
    }
    if (!param.where.has_value()) {
      num_inputs += 1;
    }
    std::vector<std::string> result;
    switch (num_inputs) {
      case 1 : result = std::vector<std::string>{"input1"}; break;
      case 2 : result = std::vector<std::string>{"input1", "input2"}; break;
      case 3 : result = std::vector<std::string>{"input1", "input2", "input3"};
      break;
    }
    return result;
  })
.set_attr_parser(ParamParser<NumpyCopytoParam>)
// .set_attr<nnvm::FInferType>("FInferType", NumpyCopytoType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyCopytoForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("input1", "NDArray-or-Symbol", "Source input")
.add_argument("input2", "NDArray-or-Symbol", "Source input")
.add_argument("input3", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyCopytoParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
