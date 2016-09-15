/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReduceAxesParam);
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
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastAxesShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_to)
.MXNET_DESCRIBE("Broadcast src to shape")
.set_attr_parser(ParamParser<BroadcastToParam>)
.set_attr<nnvm::FInferShape>("FInferShape", BroadcastToShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

}  // namespace op
}  // namespace mxnet
