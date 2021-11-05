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
 * \file sample_multinomial_op.h
 * \brief Operator for sampling from multinomial distributions
 */
#include "./sample_multinomial_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SampleMultinomialParam);

NNVM_REGISTER_OP(_sample_multinomial)
    .add_alias("sample_multinomial")
    .add_alias("_npx__random_multinomial")
    .describe(R"code(Concurrent sampling from multiple multinomial distributions.

Samples are distributed according to a multinomial distribution parametrized by
*n* (number of experiments) and *p* (success probabilities of the k possible outcomes
in each experiment). Samples will always be returned as a floating point data type.

Note that the input distribution must be normalized, i.e. *p* must sum to
1 along its last axis.

Examples::

   n = [5., 6.]
   probs = [[0., 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0.]]

   multinomial(n, probs) = [[0., 0., 0., 3., 2.],
                            [0., 3., 1., 2., 0.]]
)code")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<SampleMultinomialParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", SampleMultinomialOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SampleMultinomialOpType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const nnvm::NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{
                                      ResourceRequest::kParallelRandom,
                                      ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       std::vector<std::string> v = {"n", "p"};
                                       v.resize(2);
                                       return v;
                                     })
    .set_attr<FCompute>("FCompute<cpu>", SampleMultinomialForward<cpu>)
    .add_argument("n", "NDArray-or-Symbol", "Number of experiments")
    .add_argument("p",
                  "NDArray-or-Symbol",
                  "Probability of every outcome in each experiment. Must sum to 1 on the last axis")
    .add_arguments(SampleMultinomialParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
