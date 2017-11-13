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
 * \file elemwise_unary_op_trig.cc
 * \brief CPU Implementation of unary trigometric functions.
 */
#include <mxnet/base.h>
#include "elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"

namespace mxnet {
namespace op {

// sin
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(sin, cpu, mshadow_op::sin)
MXNET_ADD_SPARSE_OP_ALIAS(sin)
.describe(R"code(Computes the element-wise sine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

The storage type of ``sin`` output depends upon the input storage type:

   - sin(default) = default
   - sin(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sin" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_sin, unary_bwd<mshadow_op::sin_grad>);

// cos
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(cos, cpu, mshadow_op::cos)
MXNET_ADD_SPARSE_OP_ALIAS(cos)
.describe(R"code(Computes the element-wise cosine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]

The storage type of ``cos`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_cos"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_cos, unary_bwd<mshadow_op::cos_grad>);

// tan
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(tan, cpu, mshadow_op::tan)
MXNET_ADD_SPARSE_OP_ALIAS(tan)
.describe(R"code(Computes the element-wise tangent of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

The storage type of ``tan`` output depends upon the input storage type:

   - tan(default) = default
   - tan(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tan" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_tan, unary_bwd<mshadow_op::tan_grad>);

// arcsin
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(arcsin, cpu, mshadow_op::arcsin)
MXNET_ADD_SPARSE_OP_ALIAS(arcsin)
.describe(R"code(Returns element-wise inverse sine of the input array.

The input should be in the range `[-1, 1]`.
The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].

.. math::
   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]

The storage type of ``arcsin`` output depends upon the input storage type:

   - arcsin(default) = default
   - arcsin(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsin" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arcsin,
                                                  unary_bwd<mshadow_op::arcsin_grad>);

// arccos
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(arccos, cpu, mshadow_op::arccos)
MXNET_ADD_SPARSE_OP_ALIAS(arccos)
.describe(R"code(Returns element-wise inverse cosine of the input array.

The input should be in range `[-1, 1]`.
The output is in the closed interval :math:`[0, \pi]`

.. math::
   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]

The storage type of ``arccos`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccos" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arccos,
                                                  unary_bwd<mshadow_op::arccos_grad>);

// arctan
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(arctan, cpu, mshadow_op::arctan)
MXNET_ADD_SPARSE_OP_ALIAS(arctan)
.describe(R"code(Returns element-wise inverse tangent of the input array.

The output is in the closed interval :math:`[-\pi/2, \pi/2]`

.. math::
   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

The storage type of ``arctan`` output depends upon the input storage type:

   - arctan(default) = default
   - arctan(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctan" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arctan,
                                                  unary_bwd<mshadow_op::arctan_grad>);

// degrees
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(degrees, cpu, mshadow_op::degrees)
MXNET_ADD_SPARSE_OP_ALIAS(degrees)
.describe(R"code(Converts each element of the input array from radians to degrees.

.. math::
   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

The storage type of ``degrees`` output depends upon the input storage type:

   - degrees(default) = default
   - degrees(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_degrees" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_degrees,
                                                  unary_bwd<mshadow_op::degrees_grad>);

// radians
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(radians, cpu, mshadow_op::radians)
MXNET_ADD_SPARSE_OP_ALIAS(radians)
.describe(R"code(Converts each element of the input array from degrees to radians.

.. math::
   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

The storage type of ``radians`` output depends upon the input storage type:

   - radians(default) = default
   - radians(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_radians" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_radians,
                                                  unary_bwd<mshadow_op::radians_grad>);

// sinh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(sinh, cpu, mshadow_op::sinh)
MXNET_ADD_SPARSE_OP_ALIAS(sinh)
.describe(R"code(Returns the hyperbolic sine of the input array, computed element-wise.

.. math::
   sinh(x) = 0.5\times(exp(x) - exp(-x))

The storage type of ``sinh`` output depends upon the input storage type:

   - sinh(default) = default
   - sinh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sinh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_sinh, unary_bwd<mshadow_op::sinh_grad>);

// cosh
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(cosh, cpu, mshadow_op::cosh)
MXNET_ADD_SPARSE_OP_ALIAS(cosh)
.describe(R"code(Returns the hyperbolic cosine  of the input array, computed element-wise.

.. math::
   cosh(x) = 0.5\times(exp(x) + exp(-x))

The storage type of ``cosh`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_cosh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_cosh, unary_bwd<mshadow_op::cosh_grad>);

// tanh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(tanh, cpu, mshadow_op::tanh)
MXNET_ADD_SPARSE_OP_ALIAS(tanh)
.describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.

.. math::
   tanh(x) = sinh(x) / cosh(x)

The storage type of ``tanh`` output depends upon the input storage type:

   - tanh(default) = default
   - tanh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tanh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_tanh, unary_bwd<mshadow_op::tanh_grad>);

// arcsinh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(arcsinh, cpu, mshadow_op::arcsinh)
MXNET_ADD_SPARSE_OP_ALIAS(arcsinh)
.describe(R"code(Returns the element-wise inverse hyperbolic sine of the input array, \
computed element-wise.

The storage type of ``arcsinh`` output depends upon the input storage type:

   - arcsinh(default) = default
   - arcsinh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsinh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arcsinh,
                                                  unary_bwd<mshadow_op::arcsinh_grad>);

// arccosh
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(arccosh, cpu, mshadow_op::arccosh)
MXNET_ADD_SPARSE_OP_ALIAS(arccosh)
.describe(R"code(Returns the element-wise inverse hyperbolic cosine of the input array, \
computed element-wise.

The storage type of ``arccosh`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccosh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arccosh,
                                                  unary_bwd<mshadow_op::arccosh_grad>);

// arctanh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(arctanh, cpu, mshadow_op::arctanh)
MXNET_ADD_SPARSE_OP_ALIAS(arctanh)
.describe(R"code(Returns the element-wise inverse hyperbolic tangent of the input array, \
computed element-wise.

The storage type of ``arctanh`` output depends upon the input storage type:

   - arctanh(default) = default
   - arctanh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctanh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arctanh,
                                                  unary_bwd<mshadow_op::arctanh_grad>);


}  // namespace op
}  // namespace mxnet
