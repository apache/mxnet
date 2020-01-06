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
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_sum_value.cc
 * \brief CPU Implementation of broadcast and reduce sum (and related) functions based on value.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_REDUCE(sum)
MXNET_ADD_SPARSE_OP_ALIAS(sum)
.add_alias("sum_axis")
.describe(R"code(Computes the sum of array elements over given axes.

.. Note::

  `sum` and `sum_axis` are equivalent.
  For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
  Setting keepdims or exclude to True will cause a fallback to dense operator.

Example::

  data = [[[1, 2], [2, 3], [1, 3]],
          [[1, 4], [4, 3], [5, 2]],
          [[7, 1], [7, 2], [7, 3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

  data = [[1, 2, 0],
          [3, 0, 1],
          [4, 1, 0]]

  csr = cast_storage(data, 'csr')

  sum(csr, axis=0)
  [ 8.  3.  1.]

  sum(csr, axis=1)
  [ 3.  4.  5.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ReduceAxesOpForwardEx<cpu, mshadow::red::sum>)
.set_attr<FInferStorageType>("FInferStorageType", ReduceAxesOpForwardStorage)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_sum)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu>);

MXNET_OPERATOR_REGISTER_REDUCE(mean)
MXNET_ADD_SPARSE_OP_ALIAS(mean)
.describe(get_reduce_axes_description("mean", __LINE__))
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum, true>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ReduceAxesOpForwardEx<cpu, mshadow::red::sum, true>)
.set_attr<FInferStorageType>("FInferStorageType", ReduceAxesOpForwardStorage)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mean"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_mean)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseNone<cpu, true>);

MXNET_OPERATOR_REGISTER_REDUCE(nansum)
.describe(R"code(Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow_op::nansum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_nansum" });

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_nansum)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesBackwardUseInOut<cpu, mshadow_op::nansum_grad>);

}  // namespace op
}  // namespace mxnet
