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
.describe(R"code(Return the top *k* elements in an array.

Examples::

  x = [[ 0.3,  0.2,  0.4],
       [ 0.1,  0.3,  0.2]]

  // return the index of the largest element on last axis
  topk(x) = [[ 2.],
             [ 1.]]

  // return the value of the top-2 elements on last axis
  topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
                                   [ 0.3,  0.2]]

  // flatten and then return both index and value
  topk(x, ret_typ='both', k=2, axis=None) = [ 0.4,  0.3], [ 2.,  0.]

)code" ADD_FILELINE)
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
.add_argument("src", "ndarray-or-symbol", "Source input")
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
.describe(R"code(Return a sorted copy of an array.

Examples::

  x = [[ 1, 4],
       [ 3, 1]]

  // sort along the last axis
  sort(x) = [[ 1.,  4.],
             [ 1.,  3.]]

  // flatten and then sort
  sort(x, axis=None) = [ 1.,  1.,  3.,  4.]

  // sort long the first axis
  sort(x, axis=0) = [[ 1.,  1.],
                     [ 3.,  4.]]

  // in a descend order
  sort(x, is_ascend=0) = [[ 4.,  1.],
                          [ 3.,  1.]]

)code" ADD_FILELINE)
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
.add_argument("src", "ndarray-or-symbol", "Source input")
.add_arguments(SortParam::__FIELDS__());

NNVM_REGISTER_OP(argsort)
.describe(R"code(Returns the indices that can sort an array.

Examples::

  x = [[ 0.3,  0.2,  0.4],
       [ 0.1,  0.3,  0.2]]

  // sort along axis -1
  argsort(x) = [[ 1.,  0.,  2.],
                [ 0.,  2.,  1.]]

  // sort along axis 0
  argsort(x, axis=0) = [[ 1.,  0.,  1.]
                        [ 0.,  1.,  0.]]

  // flatten and then sort
  argsort(x, axis=None) = [ 3.,  1.,  5.,  0.,  4.,  2.]
)code" ADD_FILELINE)
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
.add_argument("src", "ndarray-or-symbol", "Source input")
.add_arguments(ArgSortParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
