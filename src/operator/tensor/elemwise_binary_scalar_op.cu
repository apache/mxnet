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
NNVM_REGISTER_OP(_plus_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_minus_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_rminus_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::rminus>);

NNVM_REGISTER_OP(_mul_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_div_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_rdiv_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(_backward_rdiv_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::rdiv_grad>);

NNVM_REGISTER_OP(_maximum_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_backward_maximum_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::maximum_grad>);

NNVM_REGISTER_OP(_minimum_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::minimum>);

NNVM_REGISTER_OP(_backward_minimum_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::minimum_grad>);

NNVM_REGISTER_OP(_power_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_backward_power_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::power_grad>);

NNVM_REGISTER_OP(_rpower_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::rpower>);

NNVM_REGISTER_OP(_backward_rpower_scalar)
.attr<FCompute>("FCompute<gpu>", BinaryScalarBackward<gpu, mshadow_op::rpower_grad>);

}  // namespace op
}  // namespace mxnet
