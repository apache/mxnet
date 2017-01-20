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
DMLC_REGISTER_PARAMETER(SortParam);
DMLC_REGISTER_PARAMETER(ArgSortParam);

NNVM_REGISTER_OP(topk)
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
    const TopKParam& param = nnvm::get<TopKParam>(n->attrs.parsed);
    if (param.ret_typ == topk_enum::kReturnValue || param.ret_typ == topk_enum::kReturnBoth) {
      std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.begin() + 1);
      index_t n_out = n->num_outputs();
      for (index_t i = 0; i < n_out; ++i) {
        heads.emplace_back(nnvm::NodeEntry{ n, i, 0 });
      }
      return MakeGradNode("_backward_topk", n, heads, n->attrs.dict);
    } else {
      return MakeZeroGradNodes(n, ograds);
    }
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("src", "NDArray", "Source input")
.add_arguments(TopKParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_topk)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<TopKParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TopKBackward_<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

NNVM_REGISTER_OP(sort)
.MXNET_DESCRIBE("Return a sorted copy of an array.")
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<SortParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SortShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) { return 1; })
.set_attr<FCompute>("FCompute<cpu>", Sort<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const SortParam& param = nnvm::get<SortParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.begin() + 1);
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{ n, i, 0 });
    }
    return MakeGradNode("_backward_topk", n, heads,
                         {{"axis", n->attrs.dict["axis"]},
                          {"k", "0"},
                          {"ret_typ", "value"},
                          {"is_ascend", std::to_string(param.is_ascend)}});
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("src", "NDArray", "Source input")
.add_arguments(SortParam::__FIELDS__());

NNVM_REGISTER_OP(argsort)
.MXNET_DESCRIBE("Returns the indices that would sort an array.")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ArgSortParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ArgSortShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", ArgSort<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("src", "NDArray", "Source input")
.add_arguments(ArgSortParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
