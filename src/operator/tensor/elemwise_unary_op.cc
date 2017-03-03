/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_unary_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(CastParam);

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

MXNET_OPERATOR_REGISTER_UNARY(BlockGrad)
.MXNET_DESCRIBE("Get output from a symbol and pass 0 gradient back")
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

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

NNVM_REGISTER_OP(Cast)
.add_alias("cast")
.describe(R"code(Cast to a specified type, element-wise.

For example::

   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CastParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", CastType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", CastCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_cast"})
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(CastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_cast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CastCompute<cpu>);

// negative
MXNET_OPERATOR_REGISTER_UNARY(negative)
.MXNET_DESCRIBE("Negate src")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::negation>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// abs
MXNET_OPERATOR_REGISTER_UNARY(abs)
.describe(R"code(Returns the absolute value of array elements, element-wise.

For example:
   abs([-2, 0, 3]) = [2, 0, 3]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::abs>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_abs)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sign> >);

// sign
MXNET_OPERATOR_REGISTER_UNARY(sign)
.describe(R"code(Returns the indication sign of array elements, element-wise.

For example::
   sign([-2, 0, 3]) = [-1, 0, 1]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sign>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sign"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sign)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sign_grad> >);

// round
MXNET_OPERATOR_REGISTER_UNARY(round)
.describe(R"code(Round elements of the array to the nearest integer, element-wise.

For example::
   round([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -2.,  2.,  2.,  2.]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::round>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// ceil
MXNET_OPERATOR_REGISTER_UNARY(ceil)
.describe(R"code(Return the ceiling of the input, element-wise.

For example::
   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::ceil>);

// floor
MXNET_OPERATOR_REGISTER_UNARY(floor)
.describe(R"code(Return the floor of the input, element-wise.

For example::
   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::floor>);

// rint
MXNET_OPERATOR_REGISTER_UNARY(rint)
.describe(R"code(Round elements of the array to the nearest integer, element-wise.

For example::
   rint([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -2.,  1.,  2.,  2.]

The difference to ``round`` is that ``rint`` returns ``n`` for input ``n.5``
while ``round`` returns ``n+1`` for ``n>=0``.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::rint>);

// fix
MXNET_OPERATOR_REGISTER_UNARY(fix)
.describe(R"code(Round elements of the array to the nearest integer towards
zero, element-wise.

For example::
   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::fix>);

// square
MXNET_OPERATOR_REGISTER_UNARY(square)
.describe(R"code(Calculate the square of an array, element-wise.

For example::
   square(x) = x^2

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_square)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::square_grad> >);

// sqrt
MXNET_OPERATOR_REGISTER_UNARY(sqrt)
.describe(R"code(Calculate the square-root of an array, element-wise.

For example::
   sqrt(x) = \sqrt{x}
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square_root>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sqrt)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::square_root_grad> >);

// rsqrt
MXNET_OPERATOR_REGISTER_UNARY(rsqrt)
.describe(R"code(Calculate the inverse square-root of an array, element-wise.

For example::
   rsqrt(x) = 1/\sqrt{x}
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::reciprocal_square_root>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rsqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rsqrt)
.set_attr<FCompute>("FCompute<cpu>",
  BinaryCompute<cpu, unary_bwd<mshadow_op::reciprocal_square_root_grad> >);

// exp
MXNET_OPERATOR_REGISTER_UNARY(exp)
.describe(R"code(Calculate the exponential of the array, element-wise

For example::
   exp(x) = e^x \approx 2.718^x

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::exp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_mul"});

// log
MXNET_OPERATOR_REGISTER_UNARY(log)
.describe(R"code(Natural logarithm, element-wise.

The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log10
MXNET_OPERATOR_REGISTER_UNARY(log10)
.describe(R"code(Calculate the base 10 logarithm of the array, element-wise.

``10**log10(x) = x``

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log10>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log2
MXNET_OPERATOR_REGISTER_UNARY(log2)
.describe(R"code(Calculate the base 2 logarithm of the array, element-wise.

``2**log2(x) = x``

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::log_grad> >);

// sin
MXNET_OPERATOR_REGISTER_UNARY(sin)
.describe(R"code(Trigonometric sine, element-wise.

Then input is in radians (:math:`2\pi` rad equals 360 degress).

.. math::
   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sin>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sin" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_sin)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sin_grad> >);

// log1p
MXNET_OPERATOR_REGISTER_UNARY(log1p)
.describe(R"code(Calculate ``log(1 + x)``

This function is more accurate than ``log(1 + x)``  for small ``x`` so that
:math:`1+x\approx 1`

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log1p>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log1p"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log1p)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::log1p_grad> >);

// expm1
MXNET_OPERATOR_REGISTER_UNARY(expm1)
.describe(R"code(Calculate ``exp(x) - 1``

This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::expm1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_expm1"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_expm1)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::exp> >);

// cos
MXNET_OPERATOR_REGISTER_UNARY(cos)
.describe(R"code(Cosine, element-wise.

Then input is in radians (:math:`2\pi` rad equals 360 degress).

.. math::
   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cos>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_cos"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_cos)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::cos_grad> >);

// tan
MXNET_OPERATOR_REGISTER_UNARY(tan)
.describe(R"code(Tangent, element-wise.

Then input is in radians (:math:`2\pi` rad equals 360 degress).

.. math::
   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::tan>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tan" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_tan)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::tan_grad> >);

// arcsin
MXNET_OPERATOR_REGISTER_UNARY(arcsin)
.describe(R"code(Inverse sine, element-wise.

The input should be in range :math:`[-1, 1]`.
The output is in the closed interval :math:`[-\pi/2, \pi/2]`

.. math::
   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arcsin>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsin" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arcsin)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arcsin_grad> >);

// arccos
MXNET_OPERATOR_REGISTER_UNARY(arccos)
.describe(R"code(Inverse cosine, element-wise.

The input should be in range :math:`[-1, 1]`.
The output is in the closed interval :math:`[0, \pi]`

.. math::
   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arccos>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccos" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arccos)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arccos_grad> >);

// arctan
MXNET_OPERATOR_REGISTER_UNARY(arctan)
.describe(R"code(Inverse tangent, element-wise.

The output is in the closed interval :math:`[-\pi/2, \pi/2]`

.. math::
   arccos([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arctan>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctan" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arctan)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arctan_grad> >);

// degrees
MXNET_OPERATOR_REGISTER_UNARY(degrees)
.describe(R"code(Convert angles from radians to degrees.

.. math::
   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::degrees>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_degrees" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_degrees)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::degrees_grad> >);

// radians
MXNET_OPERATOR_REGISTER_UNARY(radians)
.describe(R"code(Convert angles from degrees to radians.

.. math::
   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::radians>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_radians" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_radians)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::radians_grad> >);

// sinh
MXNET_OPERATOR_REGISTER_UNARY(sinh)
.describe(R"code(Hyperbolic sine, element-wise.

For example::
   sinh(x) = 0.5\times(exp(x) - exp(-x))

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sinh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sinh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_sinh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sinh_grad> >);

// cosh
MXNET_OPERATOR_REGISTER_UNARY(cosh)
.describe(R"code(Hyperbolic cosine, element-wise.

For example::
   cosh(x) = 0.5\times(exp(x) + exp(-x))

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cosh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_cosh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_cosh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::cosh_grad> >);

// tanh
MXNET_OPERATOR_REGISTER_UNARY(tanh)
.describe(R"code(Hyperbolic tangent element-wise.

For example::
   tanh(x) = sinh(x) / cosh(x)

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::tanh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tanh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_tanh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::tanh_grad> >);

// arcsinh
MXNET_OPERATOR_REGISTER_UNARY(arcsinh)
.describe(R"code(Inverse hyperbolic sine, element-wise.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arcsinh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsinh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arcsinh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arcsinh_grad> >);

// arccosh
MXNET_OPERATOR_REGISTER_UNARY(arccosh)
.describe(R"code(Inverse hyperbolic cosine, element-wise.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arccosh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccosh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arccosh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arccosh_grad> >);

// arctanh
MXNET_OPERATOR_REGISTER_UNARY(arctanh)
.describe(R"code(Inverse hyperbolic tangent, element-wise.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arctanh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctanh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arctanh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arctanh_grad> >);

// gamma
MXNET_OPERATOR_REGISTER_UNARY(gamma)
.MXNET_DESCRIBE("The gamma function (extension of the factorial function), element-wise")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::gamma>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gamma"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_gamma)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::gamma_grad> >);

// gammaln
MXNET_OPERATOR_REGISTER_UNARY(gammaln)
.MXNET_DESCRIBE("Log of the absolute value of the gamma function, element-wise")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::gammaln>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gammaln"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_gammaln)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::gammaln_grad> >);

}  // namespace op
}  // namespace mxnet
