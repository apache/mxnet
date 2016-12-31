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
.add_alias("sum_axis")
.MXNET_DESCRIBE("Sum src along axis. If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_sum)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu>);

MXNET_OPERATOR_REGISTER_REDUCE(prod)
.MXNET_DESCRIBE("Compute product of src along axis. "
"If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute< cpu, mshadow_op::product>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_prod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_prod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut< cpu, mshadow_op::rdiv>);

MXNET_OPERATOR_REGISTER_REDUCE(nansum)
.MXNET_DESCRIBE("Sum src along axis, ignoring NaN values. "
"If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nansum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nansum" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nansum)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nansum_grad>);

MXNET_OPERATOR_REGISTER_REDUCE(nanprod)
.MXNET_DESCRIBE("Compute product of src along axis, ignoring NaN values. "
"If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nanprod>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nanprod" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nanprod)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nanprod_grad>);

MXNET_OPERATOR_REGISTER_REDUCE(max)
.add_alias("max_axis")
.MXNET_DESCRIBE("Compute max along axis. If axis is empty, global reduction is performed")
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_max"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_max)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_REDUCE(min)
.add_alias("min_axis")
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
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>);

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
