/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_plus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_plus)
.set_attr<FCompute>("FCompute<gpu>",
                    BinaryBroadcastBackwardUseNone<gpu,
                    mshadow_op::identity, mshadow_op::identity>);

NNVM_REGISTER_OP(_ewise_plus)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_minus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_backward_minus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

NNVM_REGISTER_OP(_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::minimum>);

}  // namespace op
}  // namespace mxnet

