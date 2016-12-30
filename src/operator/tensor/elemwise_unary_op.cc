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
.MXNET_DESCRIBE("Identity mapping, copy src to output")
.add_alias("identity")
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

NNVM_REGISTER_OP(_backward_copy)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>);

// identity output as first input, but attributes are constrainted to be like rhs
NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
.set_num_inputs(2)
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FGradient>(
    "FGradient",  [](const nnvm::NodePtr& n,
                     const std::vector<nnvm::NodeEntry>& ograds) {
      auto lhs = MakeGradNode("_backward_copy", n, ograds,
                              std::unordered_map<std::string, std::string>());
      nnvm::NodePtr ng = nnvm::Node::Create();
      ng->attrs.op = nnvm::Op::Get("_zeros");
      ng->attrs.name = "zeros";
      lhs.push_back(nnvm::NodeEntry{ng, 0, 0});
      return lhs;
    });

// negative
MXNET_OPERATOR_REGISTER_UNARY(negative)
.MXNET_DESCRIBE("Negate src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::negation>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// abs
MXNET_OPERATOR_REGISTER_UNARY(abs)
.MXNET_DESCRIBE("Take absolute value of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::abs>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_abs)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sign> >);

// sign
MXNET_OPERATOR_REGISTER_UNARY(sign)
.MXNET_DESCRIBE("Take sign of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sign>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sign"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sign)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sign_grad> >);

// round
MXNET_OPERATOR_REGISTER_UNARY(round)
.MXNET_DESCRIBE("Take round of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::round>);

// ceil
MXNET_OPERATOR_REGISTER_UNARY(ceil)
.MXNET_DESCRIBE("Take ceil of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::ceil>);

// floor
MXNET_OPERATOR_REGISTER_UNARY(floor)
.MXNET_DESCRIBE("Take floor of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::floor>);

// rint
MXNET_OPERATOR_REGISTER_UNARY(rint)
.MXNET_DESCRIBE("Take round of the src to nearest integer")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::rint>);

// fix
MXNET_OPERATOR_REGISTER_UNARY(fix)
.MXNET_DESCRIBE("Take round of the src to integer nearest 0")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::fix>);

// square
MXNET_OPERATOR_REGISTER_UNARY(square)
.MXNET_DESCRIBE("Take square of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_square)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::square_grad> >);

// sqrt
MXNET_OPERATOR_REGISTER_UNARY(sqrt)
.MXNET_DESCRIBE("Take square root of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square_root>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sqrt)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::square_root_grad> >);

// rsqrt
MXNET_OPERATOR_REGISTER_UNARY(rsqrt)
.MXNET_DESCRIBE("Take reciprocal square root of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::reciprocal_square_root>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rsqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rsqrt)
.set_attr<FCompute>("FCompute<cpu>",
  BinaryCompute<cpu, unary_bwd<mshadow_op::reciprocal_square_root_grad> >);

// exp
MXNET_OPERATOR_REGISTER_UNARY(exp)
.MXNET_DESCRIBE("Take exp of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::exp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_mul"});

// log
MXNET_OPERATOR_REGISTER_UNARY(log)
.MXNET_DESCRIBE("Take log of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log10
MXNET_OPERATOR_REGISTER_UNARY(log10)
.MXNET_DESCRIBE("Take base-10 log of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log10>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log2
MXNET_OPERATOR_REGISTER_UNARY(log2)
.MXNET_DESCRIBE("Take base-2 log of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::log_grad> >);

// sin
MXNET_OPERATOR_REGISTER_UNARY(sin)
.MXNET_DESCRIBE("Take sin of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sin>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sin" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_sin)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sin_grad> >);

// log1p
MXNET_OPERATOR_REGISTER_UNARY(log1p)
.MXNET_DESCRIBE("Take `log(1 + x)` in a numerically stable way")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log1p>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log1p"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log1p)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::log1p_grad> >);

// expm1
MXNET_OPERATOR_REGISTER_UNARY(expm1)
.MXNET_DESCRIBE("Take `exp(x) - 1` in a numerically stable way")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::expm1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_expm1"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_expm1)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::exp> >);

// cos
MXNET_OPERATOR_REGISTER_UNARY(cos)
.MXNET_DESCRIBE("Take cos of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cos>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_cos"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_cos)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::cos_grad> >);

// tan
MXNET_OPERATOR_REGISTER_UNARY(tan)
.MXNET_DESCRIBE("Take tan of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::tan>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tan" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_tan)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::tan_grad> >);

// arcsin
MXNET_OPERATOR_REGISTER_UNARY(arcsin)
.MXNET_DESCRIBE("Take arcsin of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arcsin>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsin" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arcsin)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arcsin_grad> >);

// arccos
MXNET_OPERATOR_REGISTER_UNARY(arccos)
.MXNET_DESCRIBE("Take arccos of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arccos>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccos" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arccos)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arccos_grad> >);

// arctan
MXNET_OPERATOR_REGISTER_UNARY(arctan)
.MXNET_DESCRIBE("Take arctan of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arctan>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctan" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arctan)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arctan_grad> >);

// degrees
MXNET_OPERATOR_REGISTER_UNARY(degrees)
.MXNET_DESCRIBE("Take degrees of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::degrees>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_degrees" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_degrees)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::degrees_grad> >);

// radians
MXNET_OPERATOR_REGISTER_UNARY(radians)
.MXNET_DESCRIBE("Take radians of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::radians>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_radians" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_radians)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::radians_grad> >);

// sinh
MXNET_OPERATOR_REGISTER_UNARY(sinh)
.MXNET_DESCRIBE("Take sinh of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sinh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sinh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_sinh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sinh_grad> >);

// cosh
MXNET_OPERATOR_REGISTER_UNARY(cosh)
.MXNET_DESCRIBE("Take cosh of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cosh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_cosh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_cosh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::cosh_grad> >);

// tanh
MXNET_OPERATOR_REGISTER_UNARY(tanh)
.MXNET_DESCRIBE("Take tanh of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::tanh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tanh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_tanh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::tanh_grad> >);

// arcsinh
MXNET_OPERATOR_REGISTER_UNARY(arcsinh)
.MXNET_DESCRIBE("Take arcsinh of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arcsinh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsinh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arcsinh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arcsinh_grad> >);

// arccosh
MXNET_OPERATOR_REGISTER_UNARY(arccosh)
.MXNET_DESCRIBE("Take arccosh of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arccosh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccosh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arccosh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arccosh_grad> >);

// arctanh
MXNET_OPERATOR_REGISTER_UNARY(arctanh)
.MXNET_DESCRIBE("Take arctanh of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arctanh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctanh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arctanh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arctanh_grad> >);

// gamma
MXNET_OPERATOR_REGISTER_UNARY(gamma)
.MXNET_DESCRIBE("Take the gamma function (extension of the factorial function) of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::gamma>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gamma"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_gamma)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::gamma_grad> >);

// gammaln
MXNET_OPERATOR_REGISTER_UNARY(gammaln)
.MXNET_DESCRIBE("Take gammaln (log of the absolute value of gamma(x)) of the src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::gammaln>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gammaln"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_gammaln)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::gammaln_grad> >);

}  // namespace op
}  // namespace mxnet
