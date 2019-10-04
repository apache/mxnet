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
  *  Copyright (c) 2019 by Contributors
  * \file stes_op.cc
  * \Straight-through-estimators round and sign operators.
  * \author Itay Golan
  */

#include "stes_op.h"


namespace mxnet {
namespace op {

// Round STE
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(_contrib_round_ste, cpu, mshadow_op::round)
.describe(R"code(Straight-through-estimator of `round()`.

In forward pass, returns element-wise rounded value to the nearest integer of the input (same as `round()`).

In backward pass, returns gradients of ``1`` everywhere (instead of ``0`` everywhere as in `round()`):
:math:`\frac{d}{dx}{round\_ste(x)} = 1` vs. :math:`\frac{d}{dx}{round(x)} = 0`.
This is useful for quantized training.

Reference: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.

Example::
  x = round_ste([-1.5, 1.5, -1.9, 1.9, 2.7])
  x.backward()
  x = [-2.,  2., -2.,  2.,  3.]
  x.grad() = [1.,  1., 1.,  1.,  1.]

The storage type of ``round_ste`` output depends upon the input storage type:
  - round_ste(default) = default
  - round_ste(row_sparse) = row_sparse
  - round_ste(csr) = csr
)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_round_ste"});

// sign
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(_contrib_sign_ste, cpu, mshadow_op::sign)
.describe(R"code(Straight-through-estimator of `sign()`.

In forward pass, returns element-wise sign of the input (same as `sign()`).

In backward pass, returns gradients of ``1`` everywhere (instead of ``0`` everywhere as in ``sign()``):
:math:`\frac{d}{dx}{sign\_ste(x)} = 1` vs. :math:`\frac{d}{dx}{sign(x)} = 0`.
This is useful for quantized training.

Reference: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.

Example::
  x = sign_ste([-2, 0, 3])
  x.backward()
  x = [-1.,  0., 1.]
  x.grad() = [1.,  1., 1.]

The storage type of ``sign_ste`` output depends upon the input storage type:
  - round_ste(default) = default
  - round_ste(row_sparse) = row_sparse
  - round_ste(csr) = csr
)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_sign_ste"});

}  // namespace op
}  // namespace mxnet
