/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_REDUCE_AXIS(argmax)
.MXNET_DESCRIBE("Returns the indices of the maximum values along an axis.")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::maximum>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_REDUCE_AXIS(argmin)
.MXNET_DESCRIBE("Returns the indices of the minimum values along an axis.")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::minimum>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// Legacy support
NNVM_REGISTER_OP(argmax_channel)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
    ReduceAxisParam param;
    param.axis = 1;
    param.keepdims = false;
    attrs->parsed = param;
  })
.set_attr<nnvm::FInferShape>("FInferShape", ReduceAxisShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::maximum>)
.add_argument("data", "NDArray-or-Symbol", "Source input");

NNVM_REGISTER_OP(pick)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ReduceAxisParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "index"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", PickOpShape)
.set_attr<nnvm::FInferType>("FInferType", PickOpType)
.set_attr<FCompute>("FCompute<cpu>", PickOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto ret = MakeNonlossGradNode("_backward_pick", n, ograds,
                                   {n->inputs[1]}, n->attrs.dict);
    auto p = MakeNode("zeros_like", n->attrs.name + "_index_backward",
                      {n->inputs[1]}, nullptr, &n);
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_argument("index", "NDArray-or-Symbol", "Index array")
.add_arguments(ReduceAxisParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_pick)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ReduceAxisParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", PickOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
