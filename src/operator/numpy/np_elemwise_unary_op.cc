#include "../elemwise_op_common.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY(__name$, __input_name$, __kernel$)  \
NNVM_REGISTER_OP(__name$)                                                               \
.set_num_inputs(1)                                                                      \
.set_num_outputs(1)                                                                     \
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                       \
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)                           \
.set_attr<nnvm::FInplaceOption>("FInplaceOption",                                       \
  [](const NodeAttrs& attrs){                                                           \
    return std::vector<std::pair<int, int> >{{0, 0}};                                   \
  })                                                                                    \
.set_attr<nnvm::FListInputNames>("FListInputNames",                                     \
  [](const NodeAttrs& attrs) {                                                          \
    return std::vector<std::string>{__input_name$};                                     \
  })                                                                                    \
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)                        \
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, __kernel$>)       \
.add_argument(__input_name$, "NDArray-or-Symbol", "The input array.")


// negative
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_negative, "x", mshadow_op::negation)
.describe(R"code(Numerical negative, element-wise.

Example::

    negative([1.,  -1.]) = [-1.,  1.]

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// reciprocal
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_reciprocal, "x", mshadow_op::reciprocal)
.describe(R"code(Return the reciprocal of the argument, element-wise.

Example::

    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_reciprocal"});

// abs
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_absolute, "x", mshadow_op::abs)
.add_alias("_numpy_abs")
.describe(R"code(Returns element-wise absolute value of the input.

Example::

   absolute([-2, 0, 3]) = [2, 0, 3]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

// sign
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_sign, "x", mshadow_op::sign)
.describe(R"code(Returns an element-wise indication of the sign of a number.
The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.

Example::

   sign([-2, 0, 3]) = [-1, 0, 1]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sign"});

// rint
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_rint, "x", mshadow_op::rint)
.describe(R"code(Round elements of the array to the nearest integer.

Example::

   rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-2., -2., -0.,  0.,  2.,  2.,  2.]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// ceil
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_ceil, "x", mshadow_op::ceil)
.describe(R"code(Return the ceiling of the input, element-wise.

The ceil of the scalar x is the smallest integer i, such that i >= x.

Example::

   ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-1., -1., -0.,  1.,  2.,  2.,  2.]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// floor
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_floor, "x", mshadow_op::floor)
.describe(R"code(Return the floor of the input, element-wise.

The floor of the scalar x is the largest integer i, such that i <= x.

Example::

   floor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-2., -2., -1.,  0.,  1.,  1.,  2.]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// trunc
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_trunc, "x", mshadow_op::trunc)
.describe(R"code(Return the truncated value of the input, element-wise.

The truncated value of the scalar x is the nearest integer i which is closer to
zero than x is. In short, the fractional part of the signed number x is discarded.

Example::

   trunc([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-1., -1., -0.,  0.,  1.,  1.,  2.]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// fix
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_fix, "x", mshadow_op::fix)
.describe(R"code(Round to nearest integer towards zero.

Round an array of floats element-wise to nearest integer towards zero.

The rounded values are returned as floats.

Example::

   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// square
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_square, "x", mshadow_op::square)
.describe(R"code(Return the element-wise square of the input.

Example::

   square([2, 3, 4]) = [4, 9, 16]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});

// sqrt
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_sqrt, "x", mshadow_op::square_root)
.describe(R"code(Return the non-negative square-root of an array, element-wise.

Example::

   sqrt([4, 9, 16]) = [2, 3, 4]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sqrt"});

// cbrt
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_cbrt, "x", mshadow_op::cube_root)
.describe(R"code(Return the cube-root of an array, element-wise.

Example::

   cbrt([1, 8, -125]) = [1, 2, -5]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_cbrt"});

// exp
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_exp, "x", mshadow_op::exp)
.describe(R"code(Calculate the exponential of all elements in the input array.

Example::

   exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_mul"});

// log
NNVM_REGISTER_OP(_numpy_log)
.describe(R"code(Returns element-wise Natural logarithmic value of the input.

The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::LogCompute<cpu, mshadow_op::log>)
.add_argument("x", "NDArray-or-Symbol", "The input array.")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log10
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_log10, "x", mshadow_op::log10)
.describe(R"code(Returns element-wise Base-10 logarithmic value of the input.

``10**log10(x) = x``

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log10"});

// log2
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_log2, "x", mshadow_op::log2)
.describe(R"code(Returns element-wise Base-2 logarithmic value of the input.

``2**log2(x) = x``

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log2"});

// log1p
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_log1p, "x", mshadow_op::log1p)
.describe(R"code(Return the natural logarithm of one plus the input array, element-wise.

Calculates ``log(1 + x)``.

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log1p"});

// expm1
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_expm1, "x", mshadow_op::expm1)
.describe(R"code(Calculate ``exp(x) - 1`` for all elements in the array.)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_expm1"});

// logical_not
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_numpy_logical_not, "x", mshadow_op::nt)
.describe(R"code(Compute the truth value of NOT x element-wise.

Example:
  logical_not([-2., 0., 1.]) = [0., 1., 0.]

)code")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet
