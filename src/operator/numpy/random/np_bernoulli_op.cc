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
 * \file np_bernoulli_op.cc
 * \brief Operator for numpy sampling from bernoulli distributions
 */

#include "./np_bernoulli_op.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyBernoulliParam);

NNVM_REGISTER_OP(_npi_bernoulli)
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
    const NumpyBernoulliParam& param = nnvm::get<NumpyBernoulliParam>(attrs.parsed);
    int num_inputs = 1;
    if (param.logit.has_value() || param.prob.has_value()) {
      num_inputs -= 1;
    }
    return num_inputs;
  }
)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyBernoulliParam& param = nnvm::get<NumpyBernoulliParam>(attrs.parsed);
    int num_inputs = 1;
    if (param.logit.has_value() || param.prob.has_value()) {
      num_inputs -= 1;
    }
    return (num_inputs == 0) ? std::vector<std::string>() : std::vector<std::string>{"input1"};
  })
.set_attr_parser(ParamParser<NumpyBernoulliParam>)
.set_attr<mxnet::FInferShape>("FInferShape", TwoparamsDistOpShape<NumpyBernoulliParam>)
.set_attr<nnvm::FInferType>("FInferType", NumpyBernoulliOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{
        ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyBernoulliForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("input1", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyBernoulliParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
