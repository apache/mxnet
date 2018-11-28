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
 * \file index_copy.cc
 * \brief
 */
#include "./index_copy-inl.h"

namespace mxnet {
namespace op {

static bool IndexCopyType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_contrib_index_copy)
.describe(R"code(Copies the elements of a `new_tensor` into the `old_tensor`.

This operator copies the elements by selecting the indices in the order given in `index`.
The output will be a new tensor containing the rest elements of old tensor and
the copied elements of new tensor.
For example, if `index[i] == j`, then the `i` th row of `new_tensor` is copied to the
`j` th row of output.

The `index` must be a vector and it must have the same size with the `0` th dimension of
`new_tensor`. Also, the `0` th dimension of old_tensor must `>=` the `0` th dimension of
`new_tensor`, or an error will be raised.

Examples::

    x = mx.nd.zeros((5,3))
    t = mx.nd.array([[1,2,3],[4,5,6],[7,8,9]])
    index = mx.nd.array([0,4,2])

    mx.nd.contrib.index_copy(x, index, t)

    [[1. 2. 3.]
     [0. 0. 0.]
     [7. 8. 9.]
     [0. 0. 0.]
     [4. 5. 6.]]
    <NDArray 5x3 @cpu(0)>

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", IndexCopyShape)
.set_attr<nnvm::FInferType>("FInferType", IndexCopyType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_contrib_backward_index_copy"})
.set_attr<FCompute>("FCompute<cpu>", IndexCopyForward<cpu>)
.add_argument("old_tensor", "NDArray-or-Symbol", "Old tensor")
.add_argument("index_vector", "NDArray-or-Symbol", "Index vector")
.add_argument("new_tensor", "NDArray-or-Symbol", "New tensor to be copied");

NNVM_REGISTER_OP(_contrib_backward_index_copy)
.set_num_inputs(4)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", IndexCopyBackward<cpu>);

}  // namespace op
}  // namespace mxnet
