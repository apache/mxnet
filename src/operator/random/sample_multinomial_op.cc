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

DMLC_REGISTER_PARAMETER(SampleCategoricalParam);
DMLC_REGISTER_PARAMETER(SampleMultinomialParam);

NNVM_REGISTER_OP(_sample_categorical)
    .add_alias("sample_categorical")
    .add_alias("_npx__random_categorical")
    .describe(R"code(Concurrent sampling from multiple categorical distributions.

*data* is an *n* dimensional array whose last dimension has length *k*, where
*k* is the number of possible outcomes of each categorical distribution. This
operator will draw *shape* samples from each distribution. If shape is empty
one sample will be drawn from each distribution.

If *get_prob* is true, a second array containing log likelihood of the drawn
samples will also be returned. This is usually used for reinforcement learning
where you can provide reward as head gradient for this array to estimate
gradient.

Note that the input distribution must be normalized, i.e. *data* must sum to
1 along its last axis.

Examples::

   probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]

   // Draw a single sample for each distribution
   sample_categorical(probs) = [3, 0]

   // Draw a vector containing two samples for each distribution
   sample_categorical(probs, shape=(2)) = [[4, 2],
                                           [0, 0]]

   // requests log likelihood
   sample_categorical(probs, get_prob=True) = [2, 1], [0.2, 0.3]
)code")
    .set_num_inputs(1)
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const SampleCategoricalParam& param = nnvm::get<SampleCategoricalParam>(attrs.parsed);
      return param.get_prob ? 2U : 1U;
    })
    .set_attr_parser(ParamParser<SampleCategoricalParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", SampleCategoricalOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SampleCategoricalOpType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const nnvm::NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kRandom,
                                                                      ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          const SampleCategoricalParam& param = nnvm::get<SampleCategoricalParam>(n->attrs.parsed);
          if (param.get_prob) {
            return MakeGradNode("_backward_sample_categorical",
                                n,
                                {ograds[1], n->inputs[0], nnvm::NodeEntry{n, 0, 0}},
                                std::unordered_map<std::string, std::string>());
          } else {
            return MakeZeroGradNodes(n, ograds);
          }
        })
    .set_attr<FCompute>("FCompute<cpu>", SampleCategoricalForward<cpu>)
    .add_argument("data",
                  "NDArray-or-Symbol",
                  "Distribution probabilities. Must sum to one on the last axis.")
    .add_arguments(SampleCategoricalParam::__FIELDS__());

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

struct SampleCategoricalBackwardCPUKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void
  Map(int i, index_t K, index_t M, DType* ograd, DType* dist, IType* out, DType* igrad) {
    for (index_t j = 0; j < M; ++j) {
      igrad[i * K + static_cast<size_t>(out[i * M + j])] +=
          ograd[i * M + j] / dist[i * K + static_cast<size_t>(out[i * M + j])];
    }
  }
};

NNVM_REGISTER_OP(_backward_sample_categorical)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>",
                        SampleCategoricalBackward<SampleCategoricalBackwardCPUKernel, cpu>);

}  // namespace op
}  // namespace mxnet
