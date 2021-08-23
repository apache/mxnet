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
 *  Copyright (c) 2019 by Contributors
 * \file reset_arrays.cc
 * \brief setting all array element values to zeros
 * \author Moises Hernandez-Fernandez, Andrei Ivanov
 */

#include "./reset_arrays-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ResetArraysParam);

NNVM_REGISTER_OP(reset_arrays)
.describe(R"code(Set to zero multiple arrays
)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return static_cast<uint32_t>(dmlc::get<ResetArraysParam>(attrs.parsed).num_arrays);
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    const uint32_t num_args = dmlc::get<ResetArraysParam>(attrs.parsed).num_arrays;
    std::vector<uint32_t> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(i);
    }
    return ret;
  })
.set_num_outputs(0)
.set_attr_parser(ParamParser<ResetArraysParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ResetArraysShape)
.set_attr<nnvm::FInferType>("FInferType", ResetArraysType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const uint32_t num_args = dmlc::get<ResetArraysParam>(attrs.parsed).num_arrays;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("array_") + std::to_string(i));
    }
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol[]", "Arrays")
.add_arguments(ResetArraysParam::__FIELDS__());

NNVM_REGISTER_OP(reset_arrays)
.set_attr<FCompute>("FCompute<cpu>", ResetArrays<cpu>);

template<>
void ResetMemory<cpu>(void *pntr, size_t len, mshadow::Stream<cpu> *s) {
  memset(pntr, 0, len);
}

}  // namespace op
}  // namespace mxnet
