/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_unary_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
// copy
NNVM_REGISTER_OP(_copy)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

NNVM_REGISTER_OP(_backward_copy)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

// identity output as first input, but attributes are constrainted to be like rhs
NNVM_REGISTER_OP(_identity_attr_like_rhs)
.set_attr<FCompute>("FCompute<gpu>", IdentityCompute<gpu>);

// negative
NNVM_REGISTER_OP(negative)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::negation>);

// abs
NNVM_REGISTER_OP(abs)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::abs>);

NNVM_REGISTER_OP(_backward_abs)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::sign> >);

// sign
NNVM_REGISTER_OP(sign)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::sign>);

NNVM_REGISTER_OP(_backward_sign)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::sign_grad> >);

// round
NNVM_REGISTER_OP(round)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::round>);

// ceil
NNVM_REGISTER_OP(ceil)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::ceil>);

// floor
NNVM_REGISTER_OP(floor)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::floor>);

// square
NNVM_REGISTER_OP(square)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::square>);

NNVM_REGISTER_OP(_backward_square)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::square_grad> >);

// sqrt
NNVM_REGISTER_OP(sqrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::square_root>);

NNVM_REGISTER_OP(_backward_sqrt)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::square_root_grad> >);

// rsqrt
NNVM_REGISTER_OP(rsqrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::reciprocal_square_root>);

NNVM_REGISTER_OP(_backward_rsqrt)
.set_attr<FCompute>("FCompute<gpu>",
  BinaryCompute<gpu, unary_bwd<mshadow_op::reciprocal_square_root_grad> >);

// exp
NNVM_REGISTER_OP(exp)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::exp>);

// log
NNVM_REGISTER_OP(log)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::log>);

NNVM_REGISTER_OP(_backward_log)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::log_grad> >);

// log1p
NNVM_REGISTER_OP(log1p)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::log1p>);

NNVM_REGISTER_OP(_backward_log1p)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::log1p_grad> >);

// expm1
NNVM_REGISTER_OP(expm1)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::expm1>);

NNVM_REGISTER_OP(_backward_expm1)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::exp> >);

}  // namespace op
}  // namespace mxnet

