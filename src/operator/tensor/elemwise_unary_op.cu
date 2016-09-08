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
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::identity>);

// negative
NNVM_REGISTER_OP(negative)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::negation>);

// abs
NNVM_REGISTER_OP(abs)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::abs>);

NNVM_REGISTER_OP(_backward_abs)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::sign> >);

// sign
NNVM_REGISTER_OP(sign)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::sign>);

NNVM_REGISTER_OP(_backward_sign)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::sign_grad> >);

// round
NNVM_REGISTER_OP(round)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::round>);

// ceil
NNVM_REGISTER_OP(ceil)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::ceil>);

// floor
NNVM_REGISTER_OP(floor)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::floor>);

// square
NNVM_REGISTER_OP(square)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::square>);

NNVM_REGISTER_OP(_backward_square)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::square_grad> >);

// sqrt
NNVM_REGISTER_OP(sqrt)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::square_root>);

NNVM_REGISTER_OP(_backward_sqrt)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::square_root_grad> >);

// rsqrt
NNVM_REGISTER_OP(rsqrt)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::reciprocal_square_root>);

NNVM_REGISTER_OP(_backward_rsqrt)
.attr<FCompute>("FCompute<gpu>",
  BinaryCompute<gpu, unary_backward<mshadow_op::reciprocal_square_root_grad> >);

// exp
NNVM_REGISTER_OP(exp)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::exp>);

// log
NNVM_REGISTER_OP(log)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::log>);

NNVM_REGISTER_OP(_backward_log)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::log_grad> >);

// cos
NNVM_REGISTER_OP(cos)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::cos>);

NNVM_REGISTER_OP(_backward_cos)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::cos_grad> >);

// sin
NNVM_REGISTER_OP(sin)
.attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::sin>);

NNVM_REGISTER_OP(_backward_sin)
.attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_backward<mshadow_op::sin_grad> >);

}  // namespace op
}  // namespace mxnet
