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
 * Copyright (c) 2015 by Contributors
 * \file lrn.cc
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_lrn-inl.h"
#endif
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_lrn-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(LRNParam param, int dtype) {
#if MXNET_USE_MKL2017 == 1
  return new MKLLRNOp<cpu, float>(param);
#endif
  return new LocalResponseNormOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* LocalResponseNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(LRNParam);

MXNET_REGISTER_OP_PROPERTY(LRN, LocalResponseNormProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_arguments(LRNParam::__FIELDS__())
.describe(R"code(Applies local response normalization to the input.

The local response normalization layer performs "lateral inhibition" by normalizing
over local input regions.

If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
activity :math:`b_{x,y}^{i}` is given by the expression:

.. math::
   b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}

where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
number of kernels in the layer.

)code" ADD_FILELINE);

}  // namespace op
}  // namespace mxnet
