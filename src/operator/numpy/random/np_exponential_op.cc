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
 * \file np_exponential_op.cc
 * \brief Operator for numpy sampling from exponential distributions
 */

#include "./np_exponential_op.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyExponentialParam);

NNVM_REGISTER_OP(_npi_exponential)
.describe("Numpy behavior exponential")
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
    const NumpyExponentialParam& param = nnvm::get<NumpyExponentialParam>(attrs.parsed);
    int num_inputs = 1;
    if (param.scale.has_value()) {
      num_inputs -= 1;
    }
    return num_inputs;
  })
.set_num_outputs(2)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyExponentialParam& param = nnvm::get<NumpyExponentialParam>(attrs.parsed);
    int num_inputs = 1;
    if (param.scale.has_value()) {
      num_inputs -= 1;
    }
    return (num_inputs == 0) ? std::vector<std::string>() : std::vector<std::string>{"input1"};
  })
.set_attr_parser(ParamParser<NumpyExponentialParam>)
.set_attr<mxnet::FInferShape>("FInferShape", TwoparamsDistOpShape<NumpyExponentialParam>)
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs &attrs, std::vector<int> *in_attrs,  std::vector<int> *out_attrs) {
    (*out_attrs)[0] = mshadow::kFloat32;
    (*out_attrs)[1] = mshadow::kFloat32;
    return true;
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{
        ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyExponentialForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_broadcast_exponential"})
.add_argument("input1", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyExponentialParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_broadcast_exponential)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<NumpyExponentialParam>)
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
    const NumpyExponentialParam& param = nnvm::get<NumpyExponentialParam>(attrs.parsed);
    int num_inputs = 5;
    if (param.scale.has_value()) num_inputs -= 1;
    return num_inputs;
  }
)
.set_num_outputs(
  [](const nnvm::NodeAttrs& attrs) {
    const NumpyExponentialParam& param = nnvm::get<NumpyExponentialParam>(attrs.parsed);
    int num_outputs = 1;
    if (param.scale.has_value()) num_outputs -= 1;
    return num_outputs;
  }
)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", ExponentialReparamBackward<cpu>)
.add_arguments(NumpyExponentialParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
