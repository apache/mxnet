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
 * \file loss_binary_op.cc
 * \brief loss function that takes a data and label
*/
#include "./loss_binary_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(softmax_cross_entropy)
.describe(R"code(Calculate cross entropy of softmax output and one-hot label.

- This operator computes the cross entropy in two steps:
  - Applies softmax function on the input array.
  - Computes and returns the cross entropy loss between the softmax output and the labels.

- The softmax function and cross entropy loss is given by:

  - Softmax Function:

  .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

  - Cross Entropy Function:

  .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)

Example::

  x = [[1, 2, 3],
       [11, 7, 5]]

  label = [2, 0]

  softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
                [0.97962922, 0.01794253, 0.00242826]]

  softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", SoftmaxCrossEntropyShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCrossEntropyForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_softmax_cross_entropy"})
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_argument("label", "NDArray-or-Symbol", "Input label");

NNVM_REGISTER_OP(_backward_softmax_cross_entropy)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCrossEntropyBackward<cpu>);

}  // namespace op
}  // namespace mxnet
