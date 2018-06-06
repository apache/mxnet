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
 * \file l1_normalization.cc
 * \brief l1 normalization operator
*/
#include "./l1_normalization-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(L1NormalizationParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new L1NormalizationOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* L1NormalizationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(L1NormalizationParam);

MXNET_REGISTER_OP_PROPERTY(L1Normalization, L1NormalizationProp)
.describe(R"code(Normalize the input array using the L1 norm.

For 1-D NDArray, it computes::

  out = data / sum(abs(data) + eps)

For N-D NDArray, if the input array has shape (N, N, ..., N),

with ``mode`` = ``instance``, it normalizes each instance in the multidimensional
array by its L1 norm.::

  for i in 0...N
    out[i,:,:,...,:] = data[i,:,:,...,:] / sum(abs(data[i,:,:,...,:]) + eps)

with ``mode`` = ``channel``, it normalizes each channel in the array by its L1 norm.::

  for i in 0...N
    out[:,i,:,...,:] = data[:,i,:,...,:] / sum(abs(data[:,i,:,...,:])) + eps)

with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position
in the array by its L1 norm.::

  for dim in 2...N
    for i in 0...N
      out[.....,i,...] = take(out, indices=i, axis=dim) / sum(abs(out), indices=i, axis=dim) + eps)
          -dim-

Example::

  x = [[[1,2],
        [3,4]],
       [[2,2],
        [5,6]]]

  L1Normalization(x, mode='instance')
  =[[[0.1       , 0.2       ],
     [0.3       , 0.4       ]],
    [[0.13333333, 0.13333333],
     [0.33333333, 0.4       ]]]

  L1Normalization(x, mode='channel')
  =[[[0.25      , 0.33333333],
     [0.75      , 0.66666667]],
    [[0.28571429, 0.25      ],
     [0.71428571, 0.75      ]]]

  L1Normalization(x, mode='spatial')
  =[[[0.33333333, 0.66666667],
     [0.42857143, 0.57142857]],
    [[0.5       , 0.5       ],
     [0.45454545, 0.54545455]]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to normalize.")
.add_arguments(L1NormalizationParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
