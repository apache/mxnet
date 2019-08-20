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
#include "elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"
#include "../nn/mkldnn/mkldnn_ops-inl.h"

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
  ElemwiseBinaryOp::Compute<cpu, unary_bwd<mshadow_op::reciprocal_grad> >)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    // ograds[0]: dL/dxgrad
    // inputs[0]: dL/dy
    // inputs[1]: x
    // f(x) = y = 1/x
    // f'(x) = -1/x^2
    // f''(x) = 2/x^3 = -2 * (f'(x) * f(x))

    const std::unordered_map<std::string, std::string> args = {{"scalar", "-2.0"}};

    auto dydx_mul_dldy = nnvm::NodeEntry{n};  // f'(x) * head_grads
    auto dydx = MakeNode("elemwise_div", n->attrs.name + "_dydx",
                         {dydx_mul_dldy, n->inputs[0]}, nullptr, &n);
    auto fx = MakeNode("reciprocal", n->attrs.name + "_fx",
                       {n->inputs[1]}, nullptr, &n);

    auto d2ydx2_mid = MakeNode("elemwise_mul", n->attrs.name + "_d2ydx2_mid",
                               {dydx_mul_dldy, nnvm::NodeEntry{fx}}, nullptr, &n);

    auto d2ydx2 = MakeNode("_mul_scalar", n->attrs.name + "_d2ydx2",
                           {nnvm::NodeEntry{d2ydx2_mid}}, &args, &n);

    std::vector<nnvm::NodeEntry> ret;

    ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                             {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
    ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad_inp",
                             {ograds[0], nnvm::NodeEntry{d2ydx2}}, nullptr, &n));
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

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_square,
                                               unary_bwd<mshadow_op::square_grad>);

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
                                                  unary_bwd<mshadow_op::square_root_grad>);

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
  _backward_rsqrt, unary_bwd<mshadow_op::reciprocal_square_root_grad>);

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
                                                  unary_bwd<mshadow_op::cube_root_grad>);

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
.set_attr<FCompute>("FCompute<cpu>",
                    ElemwiseBinaryOp::Compute<cpu,
                      unary_bwd<mshadow_op::reciprocal_cube_root_grad>>);

}  // namespace op
}  // namespace mxnet
