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
 * \file allreduce.cc
 * \brief all reduce operator
 * \author Hang Zhang
 */

#include "./allreduce-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AllReduceOpParam);

NNVM_REGISTER_OP(AllReduce)
.describe(R"code(TODO docs
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AllReduceOpParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<AllReduceOpParam>(attrs.parsed).num_args;
    return ret;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<AllReduceOpParam>(attrs.parsed).num_args;
    return ret;
  })
.set_attr<nnvm::FInferShape>("FInferShape", AllReduceShape)
.set_attr<nnvm::FInferType>("FInferType", AllReduceType)
.set_attr<FInferStorageType>("FInferStorageType", AllReduceStorageType)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FComputeEx>("FComputeEx<cpu>", AllReduceOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_AllReduce"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  uint32_t n = dmlc::get<AllReduceOpParam>(attrs.parsed).num_args;
  std::vector<std::pair<int, int> > ret;
  for (uint32_t i = 0; i < n; i++) {
    ret.push_back(std::pair<int, int>(i, i));
  }
  return ret;
})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to allreduce");

NNVM_REGISTER_OP(_backward_AllReduce)
.set_attr_parser(ParamParser<AllReduceOpParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<AllReduceOpParam>(attrs.parsed).num_args;
    return ret;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    uint32_t ret = dmlc::get<AllReduceOpParam>(attrs.parsed).num_args;
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", AllReduceStorageType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  uint32_t n = dmlc::get<AllReduceOpParam>(attrs.parsed).num_args;
  std::vector<std::pair<int, int> > ret;
  for (uint32_t i = 0; i < n; i++) {
    ret.push_back(std::pair<int, int>(i, i));
  }
  return ret;
})
.set_attr<FComputeEx>("FComputeEx<cpu>", AllReduceOpForwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
