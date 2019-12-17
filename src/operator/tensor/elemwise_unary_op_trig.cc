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
 * \brief CPU Implementation of unary trigonometric functions.
 */
#include <mxnet/base.h>
#include "elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"
#include "../../nnvm/node_op_util.h"

namespace mxnet {
namespace op {

// sin
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(sin, cpu, mshadow_op::sin)
.describe(R"code(Computes the element-wise sine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

The storage type of ``sin`` output depends upon the input storage type:

   - sin(default) = default
   - sin(row_sparse) = row_sparse
   - sin(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sin" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_sin, unary_bwd<mshadow_op::sin_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseUseIn)
      // f(x) = sin(x)
      // f'(x) = cos(x)
      // f''(x) = -sin(x)
      auto dydx = MakeNode("cos", n->attrs.name + "_dydx",
                             {n->inputs[1]}, nullptr, &n);
      auto d2ydx2 = MakeNode("negative", n->attrs.name + "_d2ydx2",
          {nnvm::NodeEntry{
            MakeNode("sin", n->attrs.name + "_grad_grad_mid", {n->inputs[1]}, nullptr, &n)
          }}, nullptr, &n);

      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad_mid",
                                    {n->inputs[0], nnvm::NodeEntry{d2ydx2}}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;

      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
    });

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

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_cos, unary_bwd<mshadow_op::cos_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dx_grad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseUseIn)
      // f(x) = cos(x)
      // f'(x) = -sin(x)
      // f''(x) = -cos(x)
      auto dydx = MakeNode("negative", n->attrs.name + "_dydx",
          {nnvm::NodeEntry{
            MakeNode("sin", n->attrs.name + "_grad_mid", {n->inputs[1]}, nullptr, &n)
          }}, nullptr, &n);
      auto d2ydx2 = MakeNode("negative", n->attrs.name + "_d2ydx2",
          {nnvm::NodeEntry{
            MakeNode("cos", n->attrs.name + "_grad_grad_mid", {n->inputs[1]}, nullptr, &n)
          }}, nullptr, &n);

      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad_mid",
                                    {n->inputs[0], nnvm::NodeEntry{d2ydx2}}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;
      // for the backward of the _backward_cos node
      // first input is the ograd and second input is x (because ElemwiseUseIn)
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
    });


// tan
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(tan, cpu, mshadow_op::tan)
.describe(R"code(Computes the element-wise tangent of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

The storage type of ``tan`` output depends upon the input storage type:

   - tan(default) = default
   - tan(row_sparse) = row_sparse
   - tan(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tan" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_tan, unary_bwd<mshadow_op::tan_grad>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // NodeEntry{n} : y_grad * f'(x)
      // n->inputs[0] : y_grad (dL/dy)
      // n->inputs[1] : y = f(x) = tan(x) (ElemwiseGradUseOut)
      // ograds[0] : head_grads (dL/dxgrad)
      // f'(x) = sec^2(x)
      // f''(x) = 2 * f'(x) * f(x)
      //
      // Note: When building gradient graph, the backward node of n->inputs[1] will be
      // added to the graph again, therefore f`(x) will be multiplied
      // So we need to compute only -> 2 * f(x) * dL/dy_grad * y_grad
      const std::unordered_map<std::string, std::string> args = {{"scalar", "2.0"}};
      auto two_y = MakeNode("_mul_scalar", n->attrs.name + "_mul_two", {n->inputs[1]}, &args, &n);
      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "_grad_mul",
                                    {n->inputs[0], nnvm::NodeEntry{two_y}}, nullptr, &n);
      auto dydx = MakeNode("elemwise_div", n->attrs.name + "_grad_div",
                           {nnvm::NodeEntry{n}, n->inputs[0]}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
  });

// arcsin
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(arcsin, cpu, mshadow_op::arcsin)
.describe(R"code(Returns element-wise inverse sine of the input array.

The input should be in the range `[-1, 1]`.
The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].

.. math::
   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]

The storage type of ``arcsin`` output depends upon the input storage type:

   - arcsin(default) = default
   - arcsin(row_sparse) = row_sparse
   - arcsin(csr) = csr

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
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(arctan, cpu, mshadow_op::arctan)
.describe(R"code(Returns element-wise inverse tangent of the input array.

The output is in the closed interval :math:`[-\pi/2, \pi/2]`

.. math::
   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

The storage type of ``arctan`` output depends upon the input storage type:

   - arctan(default) = default
   - arctan(row_sparse) = row_sparse
   - arctan(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctan" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arctan,
                                                  unary_bwd<mshadow_op::arctan_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseGradUseIn)
      // n: dL/dy * f'(x)
      // f(x) = arctanh(x)
      // dydx = f'(x) = 1/(1+x^2)
      // f''(x) = f'(x) * f'(x) * -2 * x = (-2 * x) / (1 + x^2)^2
      // return:
      //     0: dL/dy_grad * dy/dx
      //     1: dL/dy_grad * dL/dy * f''(x)
      auto dldy = n->inputs[0];
      auto x = n->inputs[1];
      auto dldy_mul_dydx = nnvm::NodeEntry{n};
      auto op = mxnet::util::NodeOpGen{n};

      auto x_grad = op.div(dldy_mul_dydx, dldy);
      auto x_grad_square = op.square(x_grad);
      auto x_grad_square_mul_x = op.mul(x_grad_square, x);
      auto x_grad_square_mul_2_x = op.mul(-2.0, x_grad_square_mul_x);
      auto grad_grad_x = op.mul(dldy, x_grad_square_mul_2_x);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(op.mul(ograds[0], x_grad));
      ret.emplace_back(op.mul(ograds[0], grad_grad_x));
      return ret;
    });

// degrees
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(degrees, cpu, mshadow_op::degrees)
.describe(R"code(Converts each element of the input array from radians to degrees.

.. math::
   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

The storage type of ``degrees`` output depends upon the input storage type:

   - degrees(default) = default
   - degrees(row_sparse) = row_sparse
   - degrees(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_degrees" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_degrees,
                                                  unary_bwd<mshadow_op::degrees_grad>);

// radians
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(radians, cpu, mshadow_op::radians)
.describe(R"code(Converts each element of the input array from degrees to radians.

.. math::
   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

The storage type of ``radians`` output depends upon the input storage type:

   - radians(default) = default
   - radians(row_sparse) = row_sparse
   - radians(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_radians" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_radians,
                                                  unary_bwd<mshadow_op::radians_grad>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// sinh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(sinh, cpu, mshadow_op::sinh)
.describe(R"code(Returns the hyperbolic sine of the input array, computed element-wise.

.. math::
   sinh(x) = 0.5\times(exp(x) - exp(-x))

The storage type of ``sinh`` output depends upon the input storage type:

   - sinh(default) = default
   - sinh(row_sparse) = row_sparse
   - sinh(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sinh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_sinh, unary_bwd<mshadow_op::sinh_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseUseIn)
      // f(x) = sinh(x)
      // f'(x) = cosh(x)
      // f''(x) = sinh(x)
      auto dydx = MakeNode("cosh", n->attrs.name + "_dydx",
                             {n->inputs[1]}, nullptr, &n);
      auto d2ydx2 = MakeNode("sinh", n->attrs.name + "_grad_grad_mid", {n->inputs[1]}, nullptr, &n);

      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad_mid",
                                    {n->inputs[0], nnvm::NodeEntry{d2ydx2}}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;

      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
    });

// cosh
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(cosh, cpu, mshadow_op::cosh)
MXNET_ADD_SPARSE_OP_ALIAS(cosh)
.describe(R"code(Returns the hyperbolic cosine  of the input array, computed element-wise.

.. math::
   cosh(x) = 0.5\times(exp(x) + exp(-x))

The storage type of ``cosh`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_cosh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_cosh, unary_bwd<mshadow_op::cosh_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseUseIn)
      // f(x) = cosh(x)
      // f'(x) = sinh(x)
      // f''(x) = cosh(x)
      auto dydx = MakeNode("sinh", n->attrs.name + "_dydx",
                             {n->inputs[1]}, nullptr, &n);
      auto d2ydx2 = MakeNode("cosh", n->attrs.name + "_grad_grad_mid", {n->inputs[1]}, nullptr, &n);

      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad_mid",
                                    {n->inputs[0], nnvm::NodeEntry{d2ydx2}}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;

      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
    });


// tanh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(tanh, cpu, mshadow_op::tanh)
.describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.

.. math::
   tanh(x) = sinh(x) / cosh(x)

The storage type of ``tanh`` output depends upon the input storage type:

   - tanh(default) = default
   - tanh(row_sparse) = row_sparse
   - tanh(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tanh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_tanh, unary_bwd<mshadow_op::tanh_grad>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // NodeEntry{n} : y_grad * f'(x)
      // n->inputs[0] : y_grad (dL/dy)
      // n->inputs[1] : y = f(x) = tanh(x) (ElemwiseGradUseOut)
      // ograds[0] : head_grads dL/dxgrad
      // f'(x) = sech^2(x)
      // f''(x) = -2 * f'(x) * f(x)
      //
      // Note: when building gradient graph, the backward node of n->inputs[1] will be
      // added to the graph again, therefore f`(x) will be multiplied
      // So we need to compute only -> -2 * f(x) * dL/dy_grad * y_grad
      const std::unordered_map<std::string, std::string> args = {{"scalar", "-2.0"}};
      auto neg_two_y = MakeNode("_mul_scalar", n->attrs.name + "_mul_neg_two",
                                {n->inputs[1]}, &args, &n);
      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "_grad_mul",
                                    {n->inputs[0], nnvm::NodeEntry{neg_two_y}}, nullptr, &n);
      auto dydx = MakeNode("elemwise_div", n->attrs.name + "_grad_div",
                           {nnvm::NodeEntry{n}, n->inputs[0]}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
  });

// arcsinh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(arcsinh, cpu, mshadow_op::arcsinh)
.describe(R"code(Returns the element-wise inverse hyperbolic sine of the input array, \
computed element-wise.

The storage type of ``arcsinh`` output depends upon the input storage type:

   - arcsinh(default) = default
   - arcsinh(row_sparse) = row_sparse
   - arcsinh(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsinh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arcsinh,
                                                  unary_bwd<mshadow_op::arcsinh_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseGradUseIn)
      // f(x) = arcsinh(x)
      // n: f'(x) = 1/(x^2 + 1)^1/2
      // f''(x) = f'(x) * x/(x^2 + 1) = x/(x^2 + 1)^(3/2)
      // Note: x/(x^2 + 1) = x * f'(x)^2
      auto dydx = n->inputs[0];
      auto x = n->inputs[1];
      auto dydx_mul_grad_x = nnvm::NodeEntry{n};
      auto op = mxnet::util::NodeOpGen{n};

      auto grad_x = op.div(dydx_mul_grad_x, dydx);
      auto grad_x_square = op.square(grad_x);
      auto grad_x_square_mul_x = op.mul(grad_x_square, x);
      auto grad_grad_x = op.mul(dydx_mul_grad_x, grad_x_square_mul_x);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(op.mul(ograds[0], grad_x));
      ret.emplace_back(op.mul(ograds[0], grad_grad_x));
      return ret;
    });

// arccosh
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(arccosh, cpu, mshadow_op::arccosh)
MXNET_ADD_SPARSE_OP_ALIAS(arccosh)
.describe(R"code(Returns the element-wise inverse hyperbolic cosine of the input array, \
computed element-wise.

The storage type of ``arccosh`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccosh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arccosh,
                                                  unary_bwd<mshadow_op::arccosh_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseGradUseIn)
      // f(x) = arccosh(x)
      // n: f'(x) = 1/((x - 1)^1/2 * (x + 1)^1/2)
      // f''(x) = f'(x) * x/((x + 1)*(x - 1)) = x/((x-1)^1/2 * (x+1)^1/2 * (x-1) * (x+1))
      // Note: x/((x-1)*(x+1)) = x * f'(x)^2
      auto dydx = n->inputs[0];
      auto x = n->inputs[1];
      auto dydx_mul_grad_x = nnvm::NodeEntry{n};
      auto op = mxnet::util::NodeOpGen{n};

      auto grad_x = op.div(dydx_mul_grad_x, dydx);
      auto grad_x_square = op.square(grad_x);
      auto grad_x_square_mul_x = op.mul(grad_x_square, x);
      auto grad_grad_x = op.mul(dydx_mul_grad_x, grad_x_square_mul_x);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(op.mul(ograds[0], grad_x));
      ret.emplace_back(op.mul(ograds[0], grad_grad_x));
      return ret;
    });

// arctanh
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(arctanh, cpu, mshadow_op::arctanh)
.describe(R"code(Returns the element-wise inverse hyperbolic tangent of the input array, \
computed element-wise.

The storage type of ``arctanh`` output depends upon the input storage type:

   - arctanh(default) = default
   - arctanh(row_sparse) = row_sparse
   - arctanh(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctanh" });

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_arctanh,
                                                  unary_bwd<mshadow_op::arctanh_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: head_grad_grads (dL/dxgrad)
      // inputs[0]: dL/dy
      // inputs[1]: x (ElemwiseGradUseIn)
      // n: dL/dy * dy/dx
      // f(x) = arctanh(x)
      // dy/dx = f'(x) = 1/(1-x^2)
      // f''(x) = f'(x) * f'(x) * 2 * x = (2 * x) / (1 - x^2)^2
      // return:
      //     0: dL/dy_grad * dy/dx
      //     1: dL/dy_grad * dL/dy * f''(x)
      auto dldy = n->inputs[0];
      auto x = n->inputs[1];
      auto dldy_mul_dydx = nnvm::NodeEntry{n};
      auto op = mxnet::util::NodeOpGen{n};

      auto x_grad = op.div(dldy_mul_dydx, dldy);
      auto x_grad_square = op.square(x_grad);
      auto x_grad_square_mul_x = op.mul(x_grad_square, x);
      auto x_grad_square_mul_2_x = op.mul(2.0, x_grad_square_mul_x);
      auto grad_grad_x = op.mul(dldy, x_grad_square_mul_2_x);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(op.mul(ograds[0], x_grad));
      ret.emplace_back(op.mul(ograds[0], grad_grad_x));
      return ret;
    });

}  // namespace op
}  // namespace mxnet
