/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReduceAxesParam);
DMLC_REGISTER_PARAMETER(ReduceAxisParam);
DMLC_REGISTER_PARAMETER(BroadcastAxesParam);
DMLC_REGISTER_PARAMETER(BroadcastToParam);

MXNET_OPERATOR_REGISTER_REDUCE(sum)
.MXNET_DESCRIBE("Sum src along axis. If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_sum)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu>);

MXNET_OPERATOR_REGISTER_REDUCE(max)
.MXNET_DESCRIBE("Compute max along axis. If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_max"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_max)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_REDUCE(min)
.MXNET_DESCRIBE("Compute min along axis. If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::minimum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_min"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_min)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_axis)
.MXNET_DESCRIBE("Broadcast src along axis")
.set_attr_parser(ParamParser<BroadcastAxesParam>)
.add_arguments(BroadcastAxesParam::__FIELDS__())
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastAxesShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_to)
.MXNET_DESCRIBE("Broadcast src to shape")
.set_attr_parser(ParamParser<BroadcastToParam>)
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastToShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

// backward op for broadcast.
NNVM_REGISTER_OP(_broadcast_backward)
.set_attr_parser(ParamParser<ReduceAxesParam>)
.set_attr<nnvm::FBackwardOutToInIndex>(
    "FBackwardOutToInIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    })
.set_attr<nnvm::FBackwardInGradIndex>(
    "FBackwardInGradIndex", [](const NodeAttrs& attrs) {
      return std::vector<uint32_t>{0};
    })
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>);

MXNET_OPERATOR_REGISTER_REDUCE_AXIS(argmax)
.MXNET_DESCRIBE("Compute argmax")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::maximum>);

MXNET_OPERATOR_REGISTER_REDUCE_AXIS(argmin)
.MXNET_DESCRIBE("Compute argmin")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::minimum>);

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
.add_argument("src", "NDArray", "Source input");

NNVM_REGISTER_OP(norm)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<TShape> *in_attrs,
     std::vector<TShape> *out_attrs) {
    CHECK_EQ(in_attrs->size(), 1);
    CHECK_EQ(out_attrs->size(), 1);
    if ((*in_attrs)[0].ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", L2NormCompute<cpu>)
.add_argument("src", "NDArray", "Source input");

}  // namespace op
}  // namespace mxnet
