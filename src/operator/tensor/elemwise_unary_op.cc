/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_unary_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
// copy
MXNET_OPERATOR_REGISTER_UNARY(_copy)
.MXNET_DESCRIBE("Copy src to output")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::identity>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"});

// negative
MXNET_OPERATOR_REGISTER_UNARY(negative)
.MXNET_DESCRIBE("Negate src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::negation>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"negative"});

// abs
MXNET_OPERATOR_REGISTER_UNARY(abs)
.MXNET_DESCRIBE("Take absolute value of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::abs>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_abs"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_abs)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::sign> >);

// sign
MXNET_OPERATOR_REGISTER_UNARY(sign)
.MXNET_DESCRIBE("Take sign of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sign>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_sign"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sign)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::sign_grad> >);

// round
MXNET_OPERATOR_REGISTER_UNARY(round)
.MXNET_DESCRIBE("Take round of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::round>);

// ceil
MXNET_OPERATOR_REGISTER_UNARY(ceil)
.MXNET_DESCRIBE("Take ceil of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::ceil>);

// floor
MXNET_OPERATOR_REGISTER_UNARY(floor)
.MXNET_DESCRIBE("Take floor of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::floor>);

// square
MXNET_OPERATOR_REGISTER_UNARY(square)
.MXNET_DESCRIBE("Take square of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_square"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_square)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::square_grad> >);

// sqrt
MXNET_OPERATOR_REGISTER_UNARY(sqrt)
.MXNET_DESCRIBE("Take square root of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square_root>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseOut{"_backward_sqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sqrt)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::square_root_grad> >);

// rsqrt
MXNET_OPERATOR_REGISTER_UNARY(rsqrt)
.MXNET_DESCRIBE("Take reciprocal square root of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::reciprocal_square_root>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_rsqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rsqrt)
.attr<FCompute>("FCompute<cpu>",
  BinaryCompute<cpu, unary_backward<mshadow_op::reciprocal_square_root_grad> >);

// exp
MXNET_OPERATOR_REGISTER_UNARY(exp)
.MXNET_DESCRIBE("Take exp of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::exp>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseOut{"_mul"});

// log
MXNET_OPERATOR_REGISTER_UNARY(log)
.MXNET_DESCRIBE("Take log of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_log"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::log_grad> >);

// cos
MXNET_OPERATOR_REGISTER_UNARY(cos)
.MXNET_DESCRIBE("Take cos of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cos>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_cos"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_cos)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::cos_grad> >);

// sin
MXNET_OPERATOR_REGISTER_UNARY(sin)
.MXNET_DESCRIBE("Take sin of the src")
.attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sin>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_sin"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sin)
.attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_backward<mshadow_op::sin_grad> >);

}  // namespace op
}  // namespace mxnet
