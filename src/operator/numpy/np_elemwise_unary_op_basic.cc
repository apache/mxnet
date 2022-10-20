/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file np_elemwise_unary_op_basic.cc
 * \brief CPU Implementation of numpy elementwise unary function.
 */
#include <mxnet/base.h>
#include "operator/tensor/elemwise_unary_op.h"
#include "operator/tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_UNARY(_npx_relu)
    .describe(R"code(Computes rectified linear activation.

.. math::

   max(features, 0)

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::relu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_relu"});

MXNET_OPERATOR_REGISTER_UNARY(_npx_sigmoid)
    .describe(R"code(Computes sigmoid of x element-wise.

.. math::

   y = 1 / (1 + exp(-x))

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::sigmoid>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"});

NNVM_REGISTER_OP(_npi_copy)
    .describe(R"code(Return an array copy of the given object.)code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<bool>{true};
                                      })
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "The input");

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY(__name$, __input_name$, __kernel$)            \
  NNVM_REGISTER_OP(__name$)                                                               \
      .set_num_inputs(1)                                                                  \
      .set_num_outputs(1)                                                                 \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                   \
      .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)                       \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                   \
                                      [](const NodeAttrs& attrs) {                        \
                                        return std::vector<std::pair<int, int> >{{0, 0}}; \
                                      })                                                  \
      .set_attr<nnvm::FListInputNames>(                                                   \
          "FListInputNames",                                                              \
          [](const NodeAttrs& attrs) { return std::vector<std::string>{__input_name$}; }) \
      .set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, __kernel$>)              \
      .add_argument(__input_name$, "NDArray-or-Symbol", "The input array.")

inline bool MixedUnaryOpType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  if (mxnet::common::is_float(a_type)) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  } else if (a_type == mshadow::kInt32 || a_type == mshadow::kInt64) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat16);
  }
  return out_attrs->at(0) != -1;
}

#define MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(__name$, __input_name$, __kernel$) \
  NNVM_REGISTER_OP(__name$)                                                               \
      .set_num_inputs(1)                                                                  \
      .set_num_outputs(1)                                                                 \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                   \
      .set_attr<nnvm::FInferType>("FInferType", MixedUnaryOpType)                         \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                   \
                                      [](const NodeAttrs& attrs) {                        \
                                        return std::vector<std::pair<int, int> >{{0, 0}}; \
                                      })                                                  \
      .set_attr<nnvm::FListInputNames>(                                                   \
          "FListInputNames",                                                              \
          [](const NodeAttrs& attrs) { return std::vector<std::string>{__input_name$}; }) \
      .set_attr<FCompute>("FCompute<cpu>", UnaryOp::ComputeMixedType<cpu, __kernel$>)     \
      .add_argument(__input_name$, "NDArray-or-Symbol", "The input array.")

#define MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_BWD_IN(name)                                      \
  NNVM_REGISTER_OP(name)                                                                          \
      .set_num_inputs(2)                                                                          \
      .set_num_outputs(1)                                                                         \
      .set_attr<nnvm::FListInputNames>("FListInputNames",                                         \
                                       [](const NodeAttrs& attrs) {                               \
                                         return std::vector<std::string>{"lhs", "rhs"};           \
                                       })                                                         \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)                           \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                           \
                                      [](const NodeAttrs& attrs) {                                \
                                        return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}}; \
                                      })                                                          \
      .add_argument("lhs", "NDArray-or-Symbol", "first input")                                    \
      .add_argument("rhs", "NDArray-or-Symbol", "second input")

/*! \brief Binary launch */
#define MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_BWD_INOUT(name)                                   \
  NNVM_REGISTER_OP(name)                                                                          \
      .set_num_inputs(3)                                                                          \
      .set_num_outputs(1)                                                                         \
      .set_attr<nnvm::FListInputNames>("FListInputNames",                                         \
                                       [](const NodeAttrs& attrs) {                               \
                                         return std::vector<std::string>{"lhs", "rhs"};           \
                                       })                                                         \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)                           \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                           \
                                      [](const NodeAttrs& attrs) {                                \
                                        return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}}; \
                                      })                                                          \
      .add_argument("lhs", "NDArray-or-Symbol", "first input")                                    \
      .add_argument("rhs", "NDArray-or-Symbol", "second input")

#define MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_IN_CPU(__name$, __kernel$)    \
  MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_BWD_IN(__name$).set_attr<FCompute>( \
      "FCompute<cpu>", ElemwiseBinaryOp::MixedUnaryBackwardUseInCompute<cpu, __kernel$>)

#define MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(__name$, __kernel$) \
  MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_BWD_IN(__name$).set_attr<FCompute>(     \
      "FCompute<cpu>", ElemwiseBinaryOp::MixedUnaryBackwardUseInCompute<cpu, __kernel$>)

#define MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEINOUT_BWD_CPU(__name$, __kernel$) \
  MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_BWD_INOUT(__name$).set_attr<FCompute>(     \
      "FCompute<cpu>", ElemwiseBinaryOp::MixedUnaryBackwardUseInOutCompute<cpu, __kernel$>)

bool NumpyUnaryBoolOpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  if (in_attrs->at(0) == -1)
    return false;
  if (out_attrs->at(0) == -1) {
    out_attrs->at(0) = mshadow::kBool;
  } else if (out_attrs->at(0) != mshadow::kBool) {
    LOG(FATAL) << "TypeError: the `out` parameter should be a boolean array";
  }
  return true;
}

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(__name$, __input_name$, __kernel$)       \
  NNVM_REGISTER_OP(__name$)                                                               \
      .set_num_inputs(1)                                                                  \
      .set_num_outputs(1)                                                                 \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                   \
      .set_attr<nnvm::FInferType>("FInferType", NumpyUnaryBoolOpType)                     \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                   \
                                      [](const NodeAttrs& attrs) {                        \
                                        return std::vector<std::pair<int, int> >{{0, 0}}; \
                                      })                                                  \
      .set_attr<nnvm::FListInputNames>(                                                   \
          "FListInputNames",                                                              \
          [](const NodeAttrs& attrs) { return std::vector<std::string>{__input_name$}; }) \
      .set_attr<FCompute>("FCompute<cpu>", UnaryOp::ComputeLogic<cpu, __kernel$>)         \
      .add_argument(__input_name$, "NDArray-or-Symbol", "The input array.")

// negative
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_npi_negative, "x", mshadow_op::negation)
    .describe(R"code(Numerical negative, element-wise.
Example::
    negative([1.,  -1.]) = [-1.,  1.]
)code")
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// reciprocal
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_npi_reciprocal, "x", mshadow_op::reciprocal)
    .describe(R"code(Return the reciprocal of the argument, element-wise.
Example::
    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]
)code")
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_reciprocal"});

// abs
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_npi_absolute, "x", mshadow_op::abs)
    .add_alias("_npi_abs")
    .describe(R"code(Returns element-wise absolute value of the input.
Example::
   absolute([-2, 0, 3]) = [2, 0, 3]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

// sign
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_npi_sign, "x", mshadow_op::sign)
    .describe(R"code(Returns an element-wise indication of the sign of a number.
The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
Example::
   sign([-2, 0, 3]) = [-1, 0, 1]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// rint
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_rint, "x", mshadow_op::rint)
    .describe(R"code(Round elements of the array to the nearest integer.
Example::
   rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-2., -2., -0.,  0.,  2.,  2.,  2.]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// ceil
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_ceil, "x", mshadow_op::ceil)
    .describe(R"code(Return the ceiling of the input, element-wise.
The ceil of the scalar x is the smallest integer i, such that i >= x.
Example::
   ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-1., -1., -0.,  1.,  2.,  2.,  2.]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// floor
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_floor, "x", mshadow_op::floor)
    .describe(R"code(Return the floor of the input, element-wise.
The floor of the scalar x is the largest integer i, such that i <= x.
Example::
   floor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-2., -2., -1.,  0.,  1.,  1.,  2.]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// bitwise_not
NNVM_REGISTER_OP(_npi_bitwise_not)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"x"};
                                     })
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::ComputeInt<cpu, mshadow_op::bitwise_not>)
    .add_argument("x", "NDArray-or-Symbol", "The input array.")
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// trunc
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_trunc, "x", mshadow_op::trunc)
    .describe(R"code(Return the truncated value of the input, element-wise.
The truncated value of the scalar x is the nearest integer i which is closer to
zero than x is. In short, the fractional part of the signed number x is discarded.
Example::
   trunc([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) = [-1., -1., -0.,  0.,  1.,  1.,  2.]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// fix
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_fix, "x", mshadow_op::fix)
    .describe(R"code(Round to nearest integer towards zero.
Round an array of floats element-wise to nearest integer towards zero.
The rounded values are returned as floats.
Example::
   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// square
MXNET_OPERATOR_REGISTER_NUMPY_UNARY(_npi_square, "x", mshadow_op::square)
    .describe(R"code(Return the element-wise square of the input.
Example::
   square([2, 3, 4]) = [4, 9, 16]
)code" ADD_FILELINE)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", EltwiseComputeExCPU<mshadow_op::square, false>)
    .set_attr<FInferStorageType>("FInferStorageType", EltwiseStorageType)
#endif  // MXNET_USE_ONEDNN
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});

// sqrt
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_sqrt, "x", mshadow_op::square_root)
    .describe(R"code(Return the non-negative square-root of an array, element-wise.
Example::
   sqrt([4, 9, 16]) = [2, 3, 4]
)code" ADD_FILELINE)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", EltwiseComputeExCPU<mshadow_op::square_root, true>)
    .set_attr<FInferStorageType>("FInferStorageType", EltwiseStorageType)
#endif  // MXNET_USE_ONEDNN
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_sqrt"});

// cbrt
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_cbrt, "x", mshadow_op::cube_root)
    .describe(R"code(Return the cube-root of an array, element-wise.
Example::
   cbrt([1, 8, -125]) = [1, 2, -5]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_cbrt"});

// exp
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_exp, "x", mshadow_op::exp)
    .describe(R"code(Calculate the exponential of all elements in the input array.
Example::
   exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]
)code" ADD_FILELINE)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", EltwiseComputeExCPU<mshadow_op::exp, true>)
    .set_attr<FInferStorageType>("FInferStorageType", EltwiseStorageType)
#endif  // MXNET_USE_ONEDNN
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_exp"});

// log
NNVM_REGISTER_OP(_npi_log)
    .describe(R"code(Returns element-wise Natural logarithmic value of the input.
The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
)code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", MixedUnaryOpType)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"x"};
                                     })
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::ComputeMixedType<cpu, mshadow_op::log>)
    .add_argument("x", "NDArray-or-Symbol", "The input array.")
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_log"});

// log10
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_log10, "x", mshadow_op::log10)
    .describe(R"code(Returns element-wise Base-10 logarithmic value of the input.
``10**log10(x) = x``
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_log10"});

// log2
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_log2, "x", mshadow_op::log2)
    .describe(R"code(Returns element-wise Base-2 logarithmic value of the input.
``2**log2(x) = x``
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_log2"});

// log1p
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_log1p, "x", mshadow_op::log1p)
    .describe(R"code(Return the natural logarithm of one plus the input array, element-wise.
Calculates ``log(1 + x)``.
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_log1p"});

// expm1
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_expm1, "x", mshadow_op::expm1)
    .describe(R"code(Calculate ``exp(x) - 1`` for all elements in the array.)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_expm1"});

// logical_not
MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(_npi_logical_not, "x", mshadow_op::np_logical_not)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// isnan
MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(_npi_isnan, "x", mshadow_op::isnan)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// isinf
MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(_npi_isinf, "x", mshadow_op::isinf)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// isposinf
MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(_npi_isposinf, "x", mshadow_op::isposinf)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// isneginf
MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(_npi_isneginf, "x", mshadow_op::isneginf)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// isfinite
MXNET_OPERATOR_REGISTER_NUMPY_UNARY_BOOL(_npi_isfinite, "x", mshadow_op::isfinite)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// sin
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_sin, "x", mshadow_op::sin)
    .describe(R"code(Trigonometric sine, element-wise.
.. math::
   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_sin"});

// cos
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_cos, "x", mshadow_op::cos)
    .describe(R"code(Computes the element-wise cosine of the input array.
.. math::
   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_cos"});

// tan
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_tan, "x", mshadow_op::tan)
    .describe(R"code(Computes the element-wise tangent of the input array.
.. math::
   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_tan"});

// arcsin
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_arcsin, "x", mshadow_op::arcsin)
    .describe(R"code(Returns element-wise inverse sine of the input array.
.. math::
   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arcsin"});

// arccos
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_arccos, "x", mshadow_op::arccos)
    .describe(R"code(Returns element-wise inverse cosine of the input array.
The input should be in range `[-1, 1]`.
The output is in the closed interval :math:`[0, \pi]`
.. math::
   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
The storage type of ``arccos`` output is always dense
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arccos"});

// arctan
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_arctan, "x", mshadow_op::arctan)
    .describe(R"code(Returns element-wise inverse tangent of the input array.
.. math::
   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arctan"});

// degrees
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_degrees, "x", mshadow_op::degrees)
    .describe(R"code(Converts each element of the input array from radians to degrees.
.. math::
   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_degrees"});

// radians
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_radians, "x", mshadow_op::radians)
    .describe(R"code(Converts each element of the input array from degrees to radians.
.. math::
   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_radians"});

// sinh
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_sinh, "x", mshadow_op::sinh)
    .describe(R"code(Returns the hyperbolic sine of the input array, computed element-wise.
.. math::
   sinh(x) = 0.5\times(exp(x) - exp(-x))
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_sinh"});

// cosh
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_cosh, "x", mshadow_op::cosh)
    .describe(R"code(Returns the hyperbolic cosine  of the input array, computed element-wise.
.. math::
   cosh(x) = 0.5\times(exp(x) + exp(-x))
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_cosh"});

// tanh
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_tanh, "x", mshadow_op::tanh)
    .describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.
.. math::
   tanh(x) = sinh(x) / cosh(x)
)code" ADD_FILELINE)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", EltwiseComputeExCPU<mshadow_op::tanh, true>)
    .set_attr<FInferStorageType>("FInferStorageType", EltwiseStorageType)
#endif  // MXNET_USE_ONEDNN
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_tanh"});

// arcsinh
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_arcsinh, "x", mshadow_op::arcsinh)
    .describe(R"code(Returns the element-wise inverse hyperbolic sine of the input array, \
computed element-wise.
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arcsinh"});

// arccosh
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_arccosh, "x", mshadow_op::arccosh)
    .describe(R"code(Returns the element-wise inverse hyperbolic cosine of the input array, \
computed element-wise.
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arccosh"});

// arctanh
MXNET_OPERATOR_REGISTER_NUMPY_MIXED_TYPE_UNARY(_npi_arctanh, "x", mshadow_op::arctanh)
    .describe(R"code(Returns the element-wise inverse hyperbolic tangent of the input array, \
computed element-wise.
)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arctanh"});

inline bool AroundOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  CHECK(in_attrs->at(0) != mshadow::kFloat16) << "Do not support `float16` as input.\n";
  return out_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(AroundParam);

NNVM_REGISTER_OP(_npi_around)
    .set_attr_parser(ParamParser<AroundParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"x"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", AroundOpType)
    .set_attr<FCompute>("FCompute<cpu>", AroundOpForward<cpu>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("x", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(AroundParam::__FIELDS__())
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

DMLC_REGISTER_PARAMETER(NumpyNanToNumParam);

NNVM_REGISTER_OP(_npi_nan_to_num)
    .describe("" ADD_FILELINE)
    .set_attr_parser(ParamParser<NumpyNanToNumParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<FCompute>("FCompute<cpu>", NumpyNanToNumOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_npi_backward_nan_to_num"})
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyNanToNumParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_backward_nan_to_num)
    .set_attr_parser(ParamParser<NumpyNanToNumParam>)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyNanToNumOpBackward<cpu>);

NNVM_REGISTER_OP(_backward_npi_exp)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"lhs", "mid", "rhs"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
                                    })
    .add_argument("lhs", "NDArray-or-Symbol", "first input")
    .add_argument("rhs", "NDArray-or-Symbol", "second input")
    .set_attr<FCompute>(
        "FCompute<cpu>",
        ElemwiseBinaryOp::MixedUnaryBackwardUseInOutCompute<cpu, op::mshadow_op::mul>)
    .set_attr<FResourceRequest>("FResourceRequest", /* For Sparse CSR */
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_log,
                                                      unary_bwd<mshadow_op::log_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_log10,
                                                      unary_bwd<mshadow_op::log10_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_log2,
                                                      unary_bwd<mshadow_op::log2_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_log1p,
                                                      unary_bwd<mshadow_op::log1p_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_expm1,
                                                      unary_bwd<mshadow_op::exp>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEINOUT_BWD_CPU(_backward_npi_sqrt,
                                                         unary_bwd<mshadow_op::square_root_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEINOUT_BWD_CPU(_backward_npi_cbrt,
                                                         unary_bwd<mshadow_op::cube_root_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_sin,
                                                      unary_bwd<mshadow_op::sin_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_IN_CPU(_backward_npi_cos, unary_bwd<mshadow_op::cos_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEINOUT_BWD_CPU(_backward_npi_tan,
                                                         unary_bwd<mshadow_op::tan_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_arcsin,
                                                      unary_bwd<mshadow_op::arcsin_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_arccos,
                                                      unary_bwd<mshadow_op::arccos_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_arctan,
                                                      unary_bwd<mshadow_op::arctan_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_degrees,
                                                      unary_bwd<mshadow_op::degrees_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_radians,
                                                      unary_bwd<mshadow_op::radians_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_sinh,
                                                      unary_bwd<mshadow_op::sinh_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_IN_CPU(_backward_npi_cosh,
                                               unary_bwd<mshadow_op::cosh_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEINOUT_BWD_CPU(_backward_npi_tanh,
                                                         unary_bwd<mshadow_op::tanh_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_arcsinh,
                                                      unary_bwd<mshadow_op::arcsinh_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_arccosh,
                                                      unary_bwd<mshadow_op::arccosh_grad>);

MXNET_OPERATOR_REGISTER_UNARY_MIXEDTYPE_USEIN_BWD_CPU(_backward_npi_arctanh,
                                                      unary_bwd<mshadow_op::arctanh_grad>);

}  // namespace op
}  // namespace mxnet
