/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
// this will be invoked by gcc and compile GPU version
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<cpu>", Transpose<cpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>);

NNVM_REGISTER_OP(crop)
.set_attr<FCompute>("FCompute<cpu>", Crop<cpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<cpu>", Slice<cpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<cpu>", SliceGrad_<cpu>);

NNVM_REGISTER_OP(flip)
.set_attr<FCompute>("FCompute<cpu>", Flip<cpu>);

NNVM_REGISTER_OP(dot)
.set_attr<FCompute>("FCompute<cpu>", DotForward_<cpu>);

NNVM_REGISTER_OP(_backward_dot)
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>);

NNVM_REGISTER_OP(batch_dot)
.set_attr<FCompute>("FCompute<cpu>", BatchDotForward_<cpu>);

NNVM_REGISTER_OP(_backward_batch_dot)
.set_attr<FCompute>("FCompute<cpu>", BatchDotBackward_<cpu>);

}  // op
}  // mxnet
