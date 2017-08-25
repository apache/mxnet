/*!
 * Copyright (c) 2017 by Contributors
 * \file sample_multinomial_op.h
 * \brief Operator for sampling from multinomial distributions
 */
#include "./sample_multinomial_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SampleMultinomialParam);


NNVM_REGISTER_OP(sample_multinomial)
.describe(R"code(Concurrent sampling from multiple multinomial distributions.

*data* is an *n* dimensional array whose last dimension has length *k*, where
*k* is the number of possible outcomes of each multinomial distribution. This
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
   sample_multinomial(probs) = [3, 0]

   // Draw a vector containing two samples for each distribution
   sample_multinomial(probs, shape=(2)) = [[4, 2],
                                           [0, 0]]

   // requests log likelihood
   sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]
)code")
.set_num_inputs(1)
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const SampleMultinomialParam& param = nnvm::get<SampleMultinomialParam>(attrs.parsed);
    return param.get_prob ? 2U : 1U;
  })
.set_attr_parser(ParamParser<SampleMultinomialParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SampleMultinomialOpShape)
.set_attr<nnvm::FInferType>("FInferType", SampleMultinomialOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{
        ResourceRequest::kRandom, ResourceRequest::kTempSpace};
    })
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const SampleMultinomialParam& param = nnvm::get<SampleMultinomialParam>(n->attrs.parsed);
    if (param.get_prob) {
      return MakeGradNode("_backward_sample_multinomial", n,
                          {ograds[1], n->inputs[0], nnvm::NodeEntry{n, 0, 0}},
                          std::unordered_map<std::string, std::string>());
    } else {
      return MakeZeroGradNodes(n, ograds);
    }
  })
.set_attr<FCompute>("FCompute<cpu>", SampleMultinomialForward<cpu>)
.add_argument("data", "NDArray-or-Symbol",
              "Distribution probabilities. Must sum to one on the last axis.")
.add_arguments(SampleMultinomialParam::__FIELDS__());


struct SampleMultinomialBackwardCPUKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t K, index_t M,
                                  DType* ograd, DType* dist, IType* out,
                                  DType* igrad) {
    for (index_t j = 0; j < M; ++j) {
      igrad[i*K + out[i*M + j]] += ograd[i*M + j] / dist[i*K + out[i*M + j]];
    }
  }
};

NNVM_REGISTER_OP(_backward_sample_multinomial)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  SampleMultinomialBackward<SampleMultinomialBackwardCPUKernel, cpu>);

}  // namespace op
}  // namespace mxnet
