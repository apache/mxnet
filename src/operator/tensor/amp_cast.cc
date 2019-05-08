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
 * \file amp_cast.cc
 * \brief Casts used by AMP
 */

#include "./amp_cast.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AMPCastParam);
DMLC_REGISTER_PARAMETER(AMPMultiCastParam);

NNVM_REGISTER_OP(amp_cast)
.describe(R"code(Cast function between low precision float/FP32 used by AMP.

It casts only between low precision float/FP32 and does not do anything for other types.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AMPCastParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", AMPCastType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", AMPCastCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_amp_cast"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(AMPCastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_amp_cast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", AMPCastCompute<cpu>);

NNVM_REGISTER_OP(amp_multicast)
.describe(R"code(Cast function used by AMP, that casts its inputs to the common widest type.

It casts only between low precision float/FP32 and does not do anything for other types.

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_outputs);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_outputs);
  })
.set_attr_parser(ParamParser<AMPMultiCastParam>)
.set_attr<mxnet::FInferShape>("FInferShape", AMPMultiCastShape)
.set_attr<nnvm::FInferType>("FInferType", AMPMultiCastType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("data_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    int num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    std::vector<std::pair<int, int>> ret;
    for (int i = 0; i < num_args; ++i) {
      ret.emplace_back(i, i);
    }
    return ret;
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    int num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    return std::vector<bool>(num_args, true);
  })
.set_attr<FCompute>("FCompute<cpu>", AMPMultiCastCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_amp_multicast"})
.add_argument("data", "NDArray-or-Symbol[]", "Weights")
.add_arguments(AMPMultiCastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_amp_multicast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_outputs);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_outputs);
  })
.set_attr_parser(ParamParser<AMPMultiCastParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("grad_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    int num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    std::vector<std::pair<int, int>> ret;
    for (int i = 0; i < num_args; ++i) {
      ret.emplace_back(i, i);
    }
    return ret;
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    int num_args = dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
    return std::vector<bool>(num_args, true);
  })
.set_attr<FCompute>("FCompute<cpu>", AMPMultiCastCompute<cpu>)
.add_argument("grad", "NDArray-or-Symbol[]", "Gradients")
.add_arguments(AMPMultiCastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
