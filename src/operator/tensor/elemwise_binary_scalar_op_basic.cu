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
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_minus_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_rminus_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rminus>);

NNVM_REGISTER_OP(_mul_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow::op::mul>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_backward_mul_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow::op::mul>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_div_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow::op::div>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_backward_div_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow::op::div>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_rdiv_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(_backward_rdiv_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<gpu,
  mshadow_op::rdiv_grad>);

NNVM_REGISTER_OP(_mod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_mod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<
  gpu, mshadow_op::mod_grad>);

NNVM_REGISTER_OP(_rmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rmod>);

NNVM_REGISTER_OP(_backward_rmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<
  gpu, mshadow_op::rmod_grad>);

}  // namespace op
}  // namespace mxnet
