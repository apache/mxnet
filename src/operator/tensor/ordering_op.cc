/*!
 *  Copyright (c) 2016 by Contributors
 * \file ordering.cc
 * \brief CPU Implementation of the ordering operations
 */
// this will be invoked by gcc and compile CPU version
#include "./ordering_op-inl.h"
#include "./elemwise_unary_op.h"


namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(TopKParam);

NNVM_REGISTER_OP(_topk)
.MXNET_DESCRIBE("Return the top k element of an input tensor along a given axis.")
.set_num_inputs(1)
.set_num_outputs(TopKNumOutputs)
.set_attr_parser(ParamParser<TopKParam>)
.set_attr<nnvm::FInferShape>("FInferShape", TopKShape)
.set_attr<nnvm::FInferType>("FInferType", TopKType)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", TopKNumVisibleOutputs)
.set_attr<FCompute>("FCompute<cpu>", TopK<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.begin() + 1);
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{ n, i, 0 });
    }
    return MakeGradNode("_backward_topk", n, heads, n->attrs.dict);
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("src", "NDArray", "Source input")
.add_arguments(TopKParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_topk)
.set_num_inputs([](const NodeAttrs& attrs) { return TopKNumOutputs(attrs) + 1;})
.set_num_outputs(1)
.set_attr_parser(ParamParser<TopKParam>)
.set_attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
.set_attr<nnvm::FBackwardInGradIndex>("FBackwardInGradIndex",
  [](const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnBoth) {
    return std::vector<uint32_t>{0, 1};
  } else {
    return std::vector<uint32_t>{0};
  }})
.set_attr<FCompute>("FCompute<cpu>", TopKBackward_<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
