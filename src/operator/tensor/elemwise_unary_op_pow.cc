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
 * \file elemwise_unary_op_pow.cc
 * \brief CPU Implementation of elementwise power (x^k for fixed k) function.
 */
#include <mxnet/base.h>

#include "../../nnvm/node_op_util.h"
#include "./elemwise_binary_op-inl.h"
#include "elemwise_unary_op.h"

namespace mxnet {
namespace op {

// reciprocal
MXNET_OPERATOR_REGISTER_UNARY(reciprocal)
    .describe(R"code(Returns the reciprocal of the argument, element-wise.

Calculates 1/x.

Example::

    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::reciprocal>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_reciprocal"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_reciprocal)
    .set_attr<FCompute>("FCompute<cpu>",
                        ElemwiseBinaryOp::Compute<cpu, unary_bwd<mshadow_op::reciprocal_grad>>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          // ograds[0]: dL/dxgrad
          // inputs[0]: dL/dy
          // inputs[1]: x
          // f(x) = y = 1/x
          // f'(x) = -1/x^2
          // f''(x) = 2/x^3 = -2 * (f'(x) * f(x))

          const std::unordered_map<std::string, std::string> args = {{"scalar", "-2.0"}};

          auto dydx_mul_dldy = nnvm::NodeEntry{n};  // f'(x) * head_grads
          auto dydx          = MakeNode(
              "elemwise_div", n->attrs.name + "_dydx", {dydx_mul_dldy, n->inputs[0]}, nullptr, &n);
          auto fx = MakeNode("reciprocal", n->attrs.name + "_fx", {n->inputs[1]}, nullptr, &n);

          auto d2ydx2_mid = MakeNode("elemwise_mul",
                                     n->attrs.name + "_d2ydx2_mid",
                                     {dydx_mul_dldy, nnvm::NodeEntry{fx}},
                                     nullptr,
                                     &n);

          auto d2ydx2 = MakeNode(
              "_mul_scalar", n->attrs.name + "_d2ydx2", {nnvm::NodeEntry{d2ydx2_mid}}, &args, &n);

          std::vector<nnvm::NodeEntry> ret;

          ret.emplace_back(MakeNode("elemwise_mul",
                                    n->attrs.name + "_backward_grad_grad",
                                    {ograds[0], nnvm::NodeEntry{dydx}},
                                    nullptr,
                                    &n));
          ret.emplace_back(MakeNode("elemwise_mul",
                                    n->attrs.name + "_backward_grad_grad_inp",
                                    {ograds[0], nnvm::NodeEntry{d2ydx2}},
                                    nullptr,
                                    &n));
          return ret;
        });

// square
#if MSHADOW_USE_MKL == 1
MXNET_MKL_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(square, cpu, mshadow_op::square, mkl_func::square)
    .describe(R"code(Returns element-wise squared value of the input.

.. math::
   square(x) = x^2

Example::

   square([2, 3, 4]) = [4, 9, 16]

The storage type of ``square`` output depends upon the input storage type:

   - square(default) = default
   - square(row_sparse) = row_sparse
   - square(csr) = csr

)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});
#else
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(square, cpu, mshadow_op::square)
    .describe(R"code(Returns element-wise squared value of the input.

.. math::
   square(x) = x^2

Example::

   square([2, 3, 4]) = [4, 9, 16]

The storage type of ``square`` output depends upon the input storage type:

   - square(default) = default
   - square(row_sparse) = row_sparse
   - square(csr) = csr

)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});
#endif

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_square, unary_bwd<mshadow_op::square_grad>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          // ograds[0]: head_grad_grads (dL/dxgrad)
          // inputs[0]: dL/dy
          // inputs[1]: x (ElemwiseGradUseIn)
          // f(x) = y = x^2
          // f'(x) = 2*x
          // f''(x) = 2
          auto dldy          = n->inputs[0];
          auto x             = n->inputs[1];
          auto dydx_mul_dldy = nnvm::NodeEntry{n};  // f'(x) * head_grads
          auto op            = mxnet::util::NodeOpGen{n};

          auto dydx = op.div(dydx_mul_dldy, dldy);

          std::unordered_map<std::string, std::string> args = {{"scalar", "2.0"}};
          auto ones_like                                    = MakeNode(
              "ones_like", n->attrs.name + "_backward_ones_like", {n->inputs[1]}, nullptr, &n);
          auto d2ydx2          = op.mul(2.0, nnvm::NodeEntry{ones_like});
          auto d2ydx2_mul_dldy = op.mul(d2ydx2, dldy);

          std::vector<nnvm::NodeEntry> ret;

          ret.emplace_back(op.mul(ograds[0], dydx));
          ret.emplace_back(op.mul(ograds[0], d2ydx2_mul_dldy));
          return ret;
        });

// sqrt
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(sqrt, cpu, mshadow_op::square_root)
    .describe(R"code(Returns element-wise square-root value of the input.

.. math::
   \textrm{sqrt}(x) = \sqrt{x}

Example::

   sqrt([4, 9, 16]) = [2, 3, 4]

The storage type of ``sqrt`` output depends upon the input storage type:

   - sqrt(default) = default
   - sqrt(row_sparse) = row_sparse
   - sqrt(csr) = csr

)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sqrt"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_sqrt,
                                                  unary_bwd<mshadow_op::square_root_grad>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          // NodeEntry{n} : y_grad * f'(x)
          // n->inputs[0] : y_grad
          // n->inputs[1] : f(x) = x^1/2
          // ograds[0] : head_grads
          // f'(x) = 1/(2*x^1/2)
          // f''(x) = f'(x) * -1/(2*x) = -1/(4 * x^3/2)
          const std::unordered_map<std::string, std::string> mul_args = {{"scalar", "0.5"}};
          auto x   = MakeNode("square", n->attrs.name + "_cube_x", {n->inputs[1]}, nullptr, &n);
          auto r_x = MakeNode(
              "reciprocal", n->attrs.name + "_reciprocal_x", {nnvm::NodeEntry{x}}, nullptr, &n);
          auto neg_r_x = MakeNode(
              "negative", n->attrs.name + "_neg_reciprocal_x", {nnvm::NodeEntry{r_x}}, nullptr, &n);
          auto half_neg_r_cube_x = MakeNode("_mul_scalar",
                                            n->attrs.name + "_half_neg_reciprocal_x",
                                            {nnvm::NodeEntry{neg_r_x}},
                                            &mul_args,
                                            &n);
          auto grad_grad_mid     = MakeNode("elemwise_mul",
                                        n->attrs.name + "_grad_grad_mid",
                                        {nnvm::NodeEntry{half_neg_r_cube_x}, n->inputs[0]},
                                        nullptr,
                                        &n);
          auto dydx              = MakeNode("elemwise_div",
                               n->attrs.name + "_grad_div",
                               {nnvm::NodeEntry{n}, n->inputs[0]},
                               nullptr,
                               &n);

          // when building gradient graph, the backward node of n->inputs[1] will be
          // added to the graph again, therefore f`(x) will be multiplied
          std::vector<nnvm::NodeEntry> ret;
          ret.emplace_back(MakeNode("elemwise_mul",
                                    n->attrs.name + "backward_grad_grad",
                                    {ograds[0], nnvm::NodeEntry{dydx}},
                                    nullptr,
                                    &n));
          ret.emplace_back(MakeNode("elemwise_mul",
                                    n->attrs.name + "backward_grad_grad_in",
                                    {ograds[0], nnvm::NodeEntry{grad_grad_mid}},
                                    nullptr,
                                    &n));
          return ret;
        });

// rsqrt
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(rsqrt, cpu, mshadow_op::reciprocal_square_root)
MXNET_ADD_SPARSE_OP_ALIAS(rsqrt)
    .describe(R"code(Returns element-wise inverse square-root value of the input.

.. math::
   rsqrt(x) = 1/\sqrt{x}

Example::

   rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]

The storage type of ``rsqrt`` output is always dense

)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rsqrt"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(
    _backward_rsqrt,
    unary_bwd<mshadow_op::reciprocal_square_root_grad>)
    .set_attr<nnvm::FGradient>("FGradient",
                               [](const nnvm::ObjectPtr& n,
                                  const std::vector<nnvm::NodeEntry>& ograds) {
                                 // NodeEntry{n} : y_grad * f'(x)
                                 // n->inputs[0] : y_grad
                                 // n->inputs[1] : x
                                 // ograds[0] : head_grad_grads (dL/dxgrad)
                                 // f(x) = 1/(x^1/2)
                                 // f'(x) = -1/(2*x^3/2)
                                 // f''(x) = f'(x) * -3/(2*x) = 3/(4 * x^5/2)
                                 auto dydx            = n->inputs[0];
                                 auto x               = n->inputs[1];
                                 auto dydx_mul_grad_x = nnvm::NodeEntry{n};
                                 auto op              = mxnet::util::NodeOpGen{n};

                                 auto two_x                = op.mul(2.0, x);
                                 auto r_two_x              = op.reciprocal(two_x);
                                 auto neg_r_two_x          = op.negative(r_two_x);
                                 auto three_by_two_neg_r_x = op.mul(3.0, neg_r_two_x);
                                 auto x_grad_grad = op.mul(three_by_two_neg_r_x, dydx_mul_grad_x);
                                 auto x_grad      = op.div(dydx_mul_grad_x, dydx);

                                 std::vector<nnvm::NodeEntry> ret;
                                 ret.emplace_back(op.mul(ograds[0], x_grad));
                                 ret.emplace_back(op.mul(ograds[0], x_grad_grad));
                                 return ret;
                               });

// cbrt
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(cbrt, cpu, mshadow_op::cube_root)
    .describe(R"code(Returns element-wise cube-root value of the input.

.. math::
   cbrt(x) = \sqrt[3]{x}

Example::

   cbrt([1, 8, -125]) = [1, 2, -5]

The storage type of ``cbrt`` output depends upon the input storage type:

   - cbrt(default) = default
   - cbrt(row_sparse) = row_sparse
   - cbrt(csr) = csr

)code" ADD_FILELINE)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_cbrt"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_cbrt,
                                                  unary_bwd<mshadow_op::cube_root_grad>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          // NodeEntry{n} : y_grad * f'(x)
          // n->inputs[0] : y_grad
          // n->inputs[1] : f(x) = x^1/3
          // ograds[0] : head_grads
          // f'(x) = 1/(3*x^2/3)
          // f''(x) = f'(x) * -2/(3*x) = -2/(9 * x^5/3)
          const std::unordered_map<std::string, std::string> three = {{"scalar", "3.0"}};
          const std::unordered_map<std::string, std::string> two   = {{"scalar", "2.0"}};
          auto x = MakeNode("_power_scalar", n->attrs.name + "_x", {n->inputs[1]}, &three, &n);
          auto three_x =
              MakeNode("_mul_scalar", n->attrs.name + "_three_x", {nnvm::NodeEntry{x}}, &three, &n);
          auto r_three_x         = MakeNode("reciprocal",
                                    n->attrs.name + "_reciprocal_three_x",
                                    {nnvm::NodeEntry{three_x}},
                                    nullptr,
                                    &n);
          auto neg_r_three_x     = MakeNode("negative",
                                        n->attrs.name + "_neg_reciprocal_three_x",
                                        {nnvm::NodeEntry{r_three_x}},
                                        nullptr,
                                        &n);
          auto two_third_neg_r_x = MakeNode("_mul_scalar",
                                            n->attrs.name + "_two_third_neg_reciprocal_x",
                                            {nnvm::NodeEntry{neg_r_three_x}},
                                            &two,
                                            &n);
          auto grad_grad_mid     = MakeNode("elemwise_mul",
                                        n->attrs.name + "_grad_grad_mid",
                                        {nnvm::NodeEntry{two_third_neg_r_x}, n->inputs[0]},
                                        nullptr,
                                        &n);
          auto dydx              = MakeNode("elemwise_div",
                               n->attrs.name + "_grad_div",
                               {nnvm::NodeEntry{n}, n->inputs[0]},
                               nullptr,
                               &n);

          // when building gradient graph, the backward node of n->inputs[1] will be
          // added to the graph again, therefore f`(x) will be multiplied
          std::vector<nnvm::NodeEntry> ret;
          ret.emplace_back(MakeNode("elemwise_mul",
                                    n->attrs.name + "backward_grad_grad",
                                    {ograds[0], nnvm::NodeEntry{dydx}},
                                    nullptr,
                                    &n));
          ret.emplace_back(MakeNode("elemwise_mul",
                                    n->attrs.name + "backward_grad_grad_in",
                                    {ograds[0], nnvm::NodeEntry{grad_grad_mid}},
                                    nullptr,
                                    &n));
          return ret;
        });

// rcbrt
MXNET_OPERATOR_REGISTER_UNARY(rcbrt)
    .describe(R"code(Returns element-wise inverse cube-root value of the input.

.. math::
   rcbrt(x) = 1/\sqrt[3]{x}

Example::

   rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::reciprocal_cube_root>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rcbrt"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rcbrt)
    .set_attr<FCompute>(
        "FCompute<cpu>",
        ElemwiseBinaryOp::Compute<cpu, unary_bwd<mshadow_op::reciprocal_cube_root_grad>>)
    .set_attr<nnvm::FGradient>("FGradient",
                               [](const nnvm::ObjectPtr& n,
                                  const std::vector<nnvm::NodeEntry>& ograds) {
                                 // NodeEntry{n} : y_grad * f'(x)
                                 // n->inputs[0] : y_grad
                                 // n->inputs[1] : x
                                 // ograds[0] : head_grad_grads (dL/dxgrad)
                                 // f(x) = 1/(x^1/3)
                                 // f'(x) = -1/(3*x^4/3)
                                 // f''(x) = f'(x) * -4/(3*x) = 4/(9 * x^7/3)
                                 auto dydx            = n->inputs[0];
                                 auto x               = n->inputs[1];
                                 auto dydx_mul_grad_x = nnvm::NodeEntry{n};
                                 auto op              = mxnet::util::NodeOpGen{n};

                                 auto three_x               = op.mul(3.0, x);
                                 auto r_three_x             = op.reciprocal(three_x);
                                 auto neg_r_three_x         = op.negative(r_three_x);
                                 auto four_by_three_neg_r_x = op.mul(4.0, neg_r_three_x);
                                 auto x_grad_grad = op.mul(four_by_three_neg_r_x, dydx_mul_grad_x);
                                 auto x_grad      = op.div(dydx_mul_grad_x, dydx);

                                 std::vector<nnvm::NodeEntry> ret;
                                 ret.emplace_back(op.mul(ograds[0], x_grad));
                                 ret.emplace_back(op.mul(ograds[0], x_grad_grad));
                                 return ret;
                               });

}  // namespace op
}  // namespace mxnet
