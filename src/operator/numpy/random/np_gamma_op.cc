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
 * \file np_gamma_op.cc
 * \brief Operator for random sampling from gamma distribution
 */

#include "./np_gamma_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyGammaParam);

inline bool NumpyGammaOpType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  const NumpyGammaParam& param = nnvm::get<NumpyGammaParam>(attrs.parsed);
  int otype = param.dtype;
  if (otype != -1) {
    (*out_attrs)[0] = otype;
  } else {
    (*out_attrs)[0] = mshadow::kFloat32;
  }
  return true;
}

NNVM_REGISTER_OP(_npi_gamma)
.describe("Numpy behavior gamma")
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
    const NumpyGammaParam& param = nnvm::get<NumpyGammaParam>(attrs.parsed);
    int num_inputs = 2;
    if (param.shape.has_value()) num_inputs -= 1;
    if (param.scale.has_value()) num_inputs -= 1;
    return num_inputs;
  }
)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyGammaParam& param = nnvm::get<NumpyGammaParam>(attrs.parsed);
    int num_inputs = 2;
    if (param.scale.has_value()) num_inputs -= 1;
    if (param.shape.has_value()) num_inputs -= 1;
    if (num_inputs == 0) return std::vector<std::string>();
    if (num_inputs == 1) return std::vector<std::string>{"input1"};
    return std::vector<std::string>{"input1", "input2"};
  })
.set_attr_parser(ParamParser<NumpyGammaParam>)
.set_attr<mxnet::FInferShape>("FInferShape", TwoparamsDistOpShape<NumpyGammaParam>)
.set_attr<nnvm::FInferType>("FInferType", NumpyGammaOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{
        ResourceRequest::kRandom,
        ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyGammaForward<cpu, double>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("input1", "NDArray-or-Symbol", "Source input")
.add_argument("input2", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyGammaParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
