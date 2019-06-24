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
 * \file swapaxis.cc
 * \brief
 * \author Ming Zhang
*/

#include "./swapaxis-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(SwapAxisParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new SwapAxisOp<cpu, DType>(param);
  });
  return op;
}

Operator* SwapAxisProp::CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                                         std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}


DMLC_REGISTER_PARAMETER(SwapAxisParam);

MXNET_REGISTER_OP_PROPERTY(SwapAxis, SwapAxisProp)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(SwapAxisParam::__FIELDS__())
.describe(R"code(Interchanges two axes of an array.

Examples::

  x = [[1, 2, 3]])
  swapaxes(x, 0, 1) = [[ 1],
                       [ 2],
                       [ 3]]

  x = [[[ 0, 1],
        [ 2, 3]],
       [[ 4, 5],
        [ 6, 7]]]  // (2,2,2) array

 swapaxes(x, 0, 2) = [[[ 0, 4],
                       [ 2, 6]],
                      [[ 1, 5],
                       [ 3, 7]]]
)code" ADD_FILELINE);

NNVM_REGISTER_OP(SwapAxis).add_alias("swapaxes").add_alias("_npi_swapaxes");
}  // namespace op
}  // namespace mxnet
