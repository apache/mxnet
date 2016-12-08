/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_maximum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_backward_maximum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::ge>);

NNVM_REGISTER_OP(_minimum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::minimum>);

NNVM_REGISTER_OP(_backward_minimum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::le>);

NNVM_REGISTER_OP(_power_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_backward_power_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::power_grad>);

NNVM_REGISTER_OP(_rpower_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::rpower>);

NNVM_REGISTER_OP(_backward_rpower_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::rpower_grad>);

NNVM_REGISTER_OP(_hypot_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::hypot>);

NNVM_REGISTER_OP(_backward_hypot_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::hypot_grad_left>);

NNVM_REGISTER_OP(smooth_l1)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::smooth_l1_loss>);

NNVM_REGISTER_OP(_backward_smooth_l1)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::smooth_l1_gradient>);

}  // namespace op
}  // namespace mxnet
