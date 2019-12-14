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
 * Copyright (c) 2017 by Contributors
 * \file softmax.cc
 * \brief CPU Implementation of softmin
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(softmin)
.describe(R"code(Applies the softmin function.

The resulting array contains elements in the range (0,1) and the elements along the given axis sum
up to 1.

.. math::
   softmin(\mathbf{z/t})_j = \frac{e^{-z_j/t}}{\sum_{k=1}^K e^{-z_k/t}}

for :math:`j = 1, ..., K`

t is the temperature parameter in softmax function. By default, t equals 1.0

Example::

  x = [[ 1.  2.  3.]
       [ 3.  2.  1.]]

  softmin(x,axis=0) = [[ 0.88079703,  0.5,  0.11920292],
                       [ 0.11920292,  0.5,  0.88079703]]

  softmin(x,axis=1) = [[ 0.66524094,  0.24472848,  0.09003057],
                       [ 0.09003057,  0.24472848,  0.66524094]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::softmax_fwd, true>)
.set_attr<nnvm::FGradient>("FGradient", SoftmaxFGradient{"_backward_softmin"})
.set_attr<nnvm::FInferType>("FInferType", SoftmaxOpType)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "The input array.")
.add_arguments(SoftmaxParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_softmin)
.set_num_inputs(SoftmaxGradOpNumInputs)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", SoftmaxGradOpInputNames)
.set_attr<mxnet::FInferShape>("FInferShape", SoftmaxGradOpShape)
.set_attr<nnvm::FInferType>("FInferType", SoftmaxGradOpType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", SoftmaxGradOpInplaceOption)
.add_argument("args", "NDArray-or-Symbol[]", "Positional input arguments")
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, op::mshadow_op::mul,
                                                        mxnet_op::softmax_bwd, true>);

}  // namespace op
}  // namespace mxnet
