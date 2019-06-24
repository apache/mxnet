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
 * \file np_multinomial_op.h
 * \brief Operator for numpy sampling from multinomial distributions
 */
#include "./np_multinomial_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyMultinomialParam);

NNVM_REGISTER_OP(_npi_multinomial)
.describe(R"code(Draw samples from a multinomial distribution. "
"The multinomial distribution is a multivariate generalisation of the binomial distribution. "
"Take an experiment with one of p possible outcomes. "
"An example of such an experiment is throwing a dice, where the outcome can be 1 through 6. "
"Each sample drawn from the distribution represents n such experiments. "
"Its values, X_i = [X_0, X_1, ..., X_p], represent the number of times the outcome was i.
)code")
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
    const NumpyMultinomialParam& param = nnvm::get<NumpyMultinomialParam>(attrs.parsed);
    return param.pvals.has_value() ? 0U : 1U;
  }
)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyMultinomialParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyMultinomialOpShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyMultinomialOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{
        ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyMultinomialForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyMultinomialParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
