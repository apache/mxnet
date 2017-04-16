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
.describe(R"code(Returns the top *k* elements in an input array along the given axis.

Examples::

  x = [[ 0.3,  0.2,  0.4],
       [ 0.1,  0.3,  0.2]]

  // returns an index of the largest element on last axis
  topk(x) = [[ 2.],
             [ 1.]]

  // returns the value of top-2 largest elements on last axis
  topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
                                   [ 0.3,  0.2]]

  // returns the value of top-2 smallest elements on last axis
  topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
                                               [ 0.1 ,  0.2]]

  // returns the value of top-2 largest elements on axis 0
  topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
                                           [ 0.1,  0.2,  0.2]]

  // flattens and then returns list of both values and indices
  topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]

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
      std::vector<nnvm::NodeEntry> inputs;
      index_t n_out = n->num_outputs();
      for (index_t i = 0; i < n_out; ++i) {
        inputs.emplace_back(nnvm::NodeEntry{ n, i, 0 });
      }
      return MakeNonlossGradNode("_backward_topk", n, {ograds[0]}, inputs, n->attrs.dict);
    } else {
      return MakeZeroGradNodes(n, ograds);
    }
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol", "The input array")
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
.describe(R"code(Returns a sorted copy of an input array along the given axis.

Examples::

  x = [[ 1, 4],
       [ 3, 1]]

  // sorts along the last axis
  sort(x) = [[ 1.,  4.],
             [ 1.,  3.]]

  // flattens and then sorts
  sort(x) = [ 1.,  1.,  3.,  4.]

  // sorts along the first axis
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
    std::vector<nnvm::NodeEntry> inputs;
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      inputs.emplace_back(nnvm::NodeEntry{ n, i, 0 });
    }
    return MakeNonlossGradNode("_backward_topk", n, {ograds[0]}, inputs,
                               {{"axis", n->attrs.dict["axis"]},
                                {"k", "0"},
                                {"ret_typ", "value"},
                                {"is_ascend", std::to_string(param.is_ascend)}});
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol", "The input array")
.add_arguments(SortParam::__FIELDS__());

NNVM_REGISTER_OP(argsort)
.describe(R"code(Returns the indices that would sort an input array along the given axis.

This function performs sorting along the given axis and returns an array of indices having same shape
as an input array that index data in sorted order.

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
  argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]
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
.add_argument("data", "NDArray-or-Symbol", "The input array")
.add_arguments(ArgSortParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
