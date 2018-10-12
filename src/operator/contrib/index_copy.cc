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

NNVM_REGISTER_OP(_contrib_index_copy)
.describe(R"code(Copies the elements of a `new_tensor` into the `old_tensor` by 
selecting the indices in the order given in `index`. The output will be a new tensor 
contains the rest elements of old tensor and the copied elements of new tensor. 
For example, if `index[i] == j`, then the `i`th row of `new_tensor` is copied to the 
`j`th row of output.

The `index` must be a vector and it must have the same size with the `0`th dimimention of 
`new_tensor`. Also, the `0`th dimimention of old_tensor must `>=` the `0`th dimimention of 
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
.set_attr<FCompute>("FCompute<cpu>", IndexCopyCompute<cpu>)
.add_argument("old_tensor", "NDArray", "Old tensor")
.add_argument("index", "NDArray", "Index vector")
.add_argument("new_tensor", "NDArray", "New tensor to be copied");

}  // namespace op
}  // namespace mxnet
