/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_unary_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
// relu
MXNET_OPERATOR_REGISTER_UNARY(relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(features, 0)

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_relu"})
.set_attr<FCompute>("FCompute<cpu>",
    UnaryLaunch<cpu, kernel_launch_op::relu>);


MXNET_OPERATOR_REGISTER_BINARY(_backward_relu)
.set_attr<FCompute>("FCompute<cpu>",
    BinaryLaunch<cpu, kernel_launch_op::relu_grad>);


// sigmoid
MXNET_OPERATOR_REGISTER_UNARY(sigmoid)
.describe(R"code(Computes sigmoid of x element-wise.

.. math::
   y = 1 / (1 + exp(-x))

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"})
.set_attr<FCompute>("FCompute<cpu>",
    UnaryLaunch<cpu, kernel_launch_op::sigmoid>);


MXNET_OPERATOR_REGISTER_BINARY(_backward_sigmoid)
.set_attr<FCompute>("FCompute<cpu>",
    BinaryLaunch<cpu, kernel_launch_op::sigmoid_grad>);


// copy
MXNET_OPERATOR_REGISTER_UNARY(_copy)
.MXNET_DESCRIBE("Returns a copy of the input.")
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
.add_alias("stop_gradient")
.describe(R"code(Stops gradient computation.

Stops the accumulated gradient of the inputs from flowing through this operator
in the backward direction. In other words, this operator prevents the contribution
of its inputs to be taken into account for computing gradients.

Example::

  v1 = [1, 2]
  v2 = [0, 1]
  a = Variable('a')
  b = Variable('b')
  b_stop_grad = stop_gradient(3 * b)
  loss = MakeLoss(b_stop_grad + a)

  executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  executor.forward(is_train=True, a=v1, b=v2)
  executor.outputs
  [ 1.  5.]

  executor.backward()
  executor.grad_arrays
  [ 0.  0.]
  [ 1.  1.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_UNARY(make_loss)
.describe(R"code(Stops gradient computation.
.. note:: ``make_loss`` is deprecated, use ``MakeLoss``.
)code" ADD_FILELINE)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"loss"};
  })
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = MakeNode("ones_like", n->attrs.name + "_backward",
                      &(n->inputs), nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    return ret;
  });

// identity output as first input, but attributes are constrainted to be like rhs
NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 1); })
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FGradient>(
    "FGradient",  [](const nnvm::NodePtr& n,
                     const std::vector<nnvm::NodeEntry>& ograds) {
      auto lhs = MakeNonlossGradNode(
          "_backward_copy", n, ograds, {},
          std::unordered_map<std::string, std::string>());
      auto ng = MakeNode("zeros_like", n->attrs.name + "rhs_backward",
                         {n->inputs[1]}, nullptr, &n);
      lhs.push_back(nnvm::NodeEntry{ng, 0, 0});
      return lhs;
    })
.add_argument("lhs", "NDArray-or-Symbol", "First input.")
.add_argument("rhs", "NDArray-or-Symbol", "Second input.");

DMLC_REGISTER_PARAMETER(CastParam);
NNVM_REGISTER_OP(Cast)
.add_alias("cast")
.describe(R"code(Casts all elements of the input to a new type.

.. note:: ``Cast`` is deprecated. Use ``cast`` instead.

Example::

   cast([0.9, 1.3], dtype='int32') = [0, 1]
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
.add_argument("data", "NDArray-or-Symbol", "The input.")
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
.describe(R"code(Returns element-wise absolute value of the input.

Example::

   abs([-2, 0, 3]) = [2, 0, 3]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::abs>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_abs)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sign> >);

// sign
MXNET_OPERATOR_REGISTER_UNARY(sign)
.describe(R"code(Returns element-wise sign of the input.

Example::

   sign([-2, 0, 3]) = [-1, 0, 1]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sign>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sign"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sign)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sign_grad> >);

// round
MXNET_OPERATOR_REGISTER_UNARY(round)
.describe(R"code(Returns element-wise rounded value to the nearest integer of the input.

Example::

   round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::round>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// rint
MXNET_OPERATOR_REGISTER_UNARY(rint)
.describe(R"code(Returns element-wise rounded value to the nearest integer of the input.

.. note::
   - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
   - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.

Example::

   rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::rint>);

// ceil
MXNET_OPERATOR_REGISTER_UNARY(ceil)
.describe(R"code(Returns element-wise ceiling of the input.

The ceil of the scalar x is the smallest integer i, such that i >= x.

Example::

   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::ceil>);

// floor
MXNET_OPERATOR_REGISTER_UNARY(floor)
.describe(R"code(Returns element-wise floor of the input.

The floor of the scalar x is the largest integer i, such that i <= x.

Example::

   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::floor>);

// trunc
MXNET_OPERATOR_REGISTER_UNARY(trunc)
.describe(R"code(Return the element-wise truncated value of the input.

The truncated value of the scalar x is the nearest integer i which is closer to 
zero than x is. In short, the fractional part of the signed number x is discarded.

Example::

   trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::trunc>);

// fix
MXNET_OPERATOR_REGISTER_UNARY(fix)
.describe(R"code(Returns element-wise rounded value to the nearest integer towards zero of the input.

Example::

   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::fix>);

// square
MXNET_OPERATOR_REGISTER_UNARY(square)
.describe(R"code(Returns element-wise squared value of the input.

.. math::
   square(x) = x^2

Example::

   square([2, 3, 4]) = [3, 9, 16]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_square)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::square_grad> >);

// sqrt
MXNET_OPERATOR_REGISTER_UNARY(sqrt)
.describe(R"code(Returns element-wise square-root value of the input.

.. math::
   \textrm{sqrt}(x) = \sqrt{x}

Example::

   sqrt([4, 9, 16]) = [2, 3, 4]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::square_root>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_sqrt)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::square_root_grad> >);

// rsqrt
MXNET_OPERATOR_REGISTER_UNARY(rsqrt)
.describe(R"code(Returns element-wise inverse square-root value of the input.

.. math::
   rsqrt(x) = 1/\sqrt{x}

Example::

   rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::reciprocal_square_root>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rsqrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rsqrt)
.set_attr<FCompute>("FCompute<cpu>",
  BinaryCompute<cpu, unary_bwd<mshadow_op::reciprocal_square_root_grad> >);

// exp
MXNET_OPERATOR_REGISTER_UNARY(exp)
.describe(R"code(Returns element-wise exponential value of the input.

.. math::
   exp(x) = e^x \approx 2.718^x

Example::

   exp([0, 1, 2]) = [inf, 1, 0.707]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::exp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_mul"});

// log
MXNET_OPERATOR_REGISTER_UNARY(log)
.describe(R"code(Returns element-wise Natural logarithmic value of the input.

The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log10
MXNET_OPERATOR_REGISTER_UNARY(log10)
.describe(R"code(Returns element-wise Base-10 logarithmic value of the input.

``10**log10(x) = x``

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log10>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log2
MXNET_OPERATOR_REGISTER_UNARY(log2)
.describe(R"code(Returns element-wise Base-2 logarithmic value of the input.

``2**log2(x) = x``

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::log_grad> >);

// sin
MXNET_OPERATOR_REGISTER_UNARY(sin)
.describe(R"code(Computes the element-wise sine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sin>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sin" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_sin)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sin_grad> >);

// log1p
MXNET_OPERATOR_REGISTER_UNARY(log1p)
.describe(R"code(Returns element-wise ``log(1 + x)`` value of the input.

This function is more accurate than ``log(1 + x)``  for small ``x`` so that
:math:`1+x\approx 1`

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::log1p>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log1p"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_log1p)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::log1p_grad> >);

// expm1
MXNET_OPERATOR_REGISTER_UNARY(expm1)
.describe(R"code(Returns ``exp(x) - 1`` computed element-wise on the input.

This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::expm1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_expm1"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_expm1)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::exp> >);

// cos
MXNET_OPERATOR_REGISTER_UNARY(cos)
.describe(R"code(Computes the element-wise cosine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cos>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_cos"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_cos)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::cos_grad> >);

// tan
MXNET_OPERATOR_REGISTER_UNARY(tan)
.describe(R"code(Computes the element-wise tangent of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::tan>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tan" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_tan)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::tan_grad> >);

// arcsin
MXNET_OPERATOR_REGISTER_UNARY(arcsin)
.describe(R"code(Returns element-wise inverse sine of the input array.

The input should be in the range `[-1, 1]`.
The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].

.. math::
   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arcsin>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsin" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arcsin)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arcsin_grad> >);

// arccos
MXNET_OPERATOR_REGISTER_UNARY(arccos)
.describe(R"code(Returns element-wise inverse cosine of the input array.

The input should be in range `[-1, 1]`.
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
.describe(R"code(Returns element-wise inverse tangent of the input array.

The output is in the closed interval :math:`[-\pi/2, \pi/2]`

.. math::
   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arctan>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctan" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arctan)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arctan_grad> >);

// degrees
MXNET_OPERATOR_REGISTER_UNARY(degrees)
.describe(R"code(Converts each element of the input array from radians to degrees.

.. math::
   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::degrees>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_degrees" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_degrees)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::degrees_grad> >);

// radians
MXNET_OPERATOR_REGISTER_UNARY(radians)
.describe(R"code(Converts each element of the input array from degrees to radians.

.. math::
   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::radians>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_radians" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_radians)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::radians_grad> >);

// sinh
MXNET_OPERATOR_REGISTER_UNARY(sinh)
.describe(R"code(Returns the hyperbolic sine of the input array, computed element-wise.

.. math::
   sinh(x) = 0.5\times(exp(x) - exp(-x))

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::sinh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sinh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_sinh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::sinh_grad> >);

// cosh
MXNET_OPERATOR_REGISTER_UNARY(cosh)
.describe(R"code(Returns the hyperbolic cosine  of the input array, computed element-wise.

.. math::
   cosh(x) = 0.5\times(exp(x) + exp(-x))

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::cosh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_cosh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_cosh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::cosh_grad> >);

// tanh
MXNET_OPERATOR_REGISTER_UNARY(tanh)
.describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.

.. math::
   tanh(x) = sinh(x) / cosh(x)

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::tanh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tanh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_tanh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::tanh_grad> >);

// arcsinh
MXNET_OPERATOR_REGISTER_UNARY(arcsinh)
.describe(R"code(Returns the element-wise inverse hyperbolic sine of the input array, computed element-wise.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arcsinh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsinh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arcsinh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arcsinh_grad> >);

// arccosh
MXNET_OPERATOR_REGISTER_UNARY(arccosh)
.describe(R"code(Returns the element-wise inverse hyperbolic cosine of the input array, computed element-wise.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arccosh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccosh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arccosh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arccosh_grad> >);

// arctanh
MXNET_OPERATOR_REGISTER_UNARY(arctanh)
.describe(R"code(Returns the element-wise inverse hyperbolic tangent of the input array, computed element-wise.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::arctanh>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctanh" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_arctanh)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::arctanh_grad> >);

// gamma
MXNET_OPERATOR_REGISTER_UNARY(gamma)
.MXNET_DESCRIBE("Returns the gamma function (extension of the factorial function to the reals)"
  " , computed element-wise on the input array.")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::gamma>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gamma"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_gamma)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::gamma_grad> >);

// gammaln
MXNET_OPERATOR_REGISTER_UNARY(gammaln)
.MXNET_DESCRIBE("Returns element-wise log of the absolute value of the gamma function"
  " of the input.")
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::gammaln>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gammaln"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_gammaln)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, unary_bwd<mshadow_op::gammaln_grad> >);

}  // namespace op
}  // namespace mxnet
