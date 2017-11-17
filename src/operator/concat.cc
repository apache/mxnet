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
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_concat-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConcatParam param, int dtype, std::vector<TShape> *in_shape) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  // MKL supports 4D input tensors only for concat operation
  // 2D/3D input tensors are reshaped to 4D in mkl_concat-inl.h
  // hence MKL supports 2D/3D/4D input tensors for concat operation
  size_t dims = (*in_shape)[0].ndim();
  bool supportedDim = (dims >= 2 && dims <= 4);
  if ((1 == param.dim) && supportedDim &&
    (param.num_args < (dnnResourceMultipleDst - dnnResourceMultipleSrc))) {
    switch (dtype) {
      case mshadow::kFloat32:
      return new MKLConcatOp<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConcatOp<cpu, double>(param);
    default:
      break;
    }
  }
  if (enableMKLWarnGenerated())
    LOG(INFO) << MKLConcatOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new ConcatOp<cpu, DType>(param);
  });
  return op;
}

Operator* ConcatProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0), in_shape);
}

DMLC_REGISTER_PARAMETER(ConcatParam);

MXNET_REGISTER_OP_PROPERTY(Concat, ConcatProp)
.describe(R"code(Joins input arrays along a given axis.

.. note:: `Concat` is deprecated. Use `concat` instead.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.

Example::

   x = [[1,1],[2,2]]
   y = [[3,3],[4,4],[5,5]]
   z = [[6,6], [7,7],[8,8]]

   concat(x,y,z,dim=0) = [[ 1.,  1.],
                          [ 2.,  2.],
                          [ 3.,  3.],
                          [ 4.,  4.],
                          [ 5.,  5.],
                          [ 6.,  6.],
                          [ 7.,  7.],
                          [ 8.,  8.]]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                         [ 4.,  4.,  7.,  7.],
                         [ 5.,  5.,  8.,  8.]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__())
.set_key_var_num_args("num_args");

NNVM_REGISTER_OP(Concat).add_alias("concat");

}  // namespace op
}  // namespace mxnet
