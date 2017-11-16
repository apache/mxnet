/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpForward<gpu>);

NNVM_REGISTER_OP(_backward_slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpBackward<gpu>);

NNVM_REGISTER_OP(_slice_assign)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignOpForward<gpu>);

NNVM_REGISTER_OP(_slice_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignScalarOpForward<gpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxis<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxisGrad_<gpu>);

NNVM_REGISTER_OP(clip)
.set_attr<FCompute>("FCompute<gpu>", Clip<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ClipEx<gpu>);

NNVM_REGISTER_OP(_backward_clip)
.set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);

NNVM_REGISTER_OP(reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(_backward_reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_backward_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpBackward<gpu>);
}  // namespace op
}  // namespace mxnet
