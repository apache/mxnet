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
 * \file broadcast_reduce_op_value.cc
 * \brief CPU Implementation of broadcast and reduce functions based on value.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReduceAxesParam);
DMLC_REGISTER_PARAMETER(ReduceAxisParam);
DMLC_REGISTER_PARAMETER(BroadcastAxesParam);
DMLC_REGISTER_PARAMETER(BroadcastToParam);
DMLC_REGISTER_PARAMETER(BroadcastLikeParam);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_axis)
.add_alias("broadcast_axes")
.describe(R"code(Broadcasts the input array over particular axes.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

`broadcast_axes` is an alias to the function `broadcast_axis`.

Example::

   // given x of shape (1,2,1)
   x = [[[ 1.],
         [ 2.]]]

   // broadcast x on on axis 2
   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                         [ 2.,  2.,  2.]]]
   // broadcast x on on axes 0 and 2
   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]],
                                                [[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastAxesParam>)
.add_arguments(BroadcastAxesParam::__FIELDS__())
.set_attr<mxnet::FInferShape>("FInferShape", BroadcastAxesShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

MXNET_OPERATOR_REGISTER_BROADCAST(broadcast_to)
.describe(R"code(Broadcasts the input array to a new shape.

Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                           [ 1.,  2.,  3.]])

The dimension which you do not want to change can also be kept as `0` which means copy the original value.
So with `shape=(2,0)`, we will obtain the same result as in the above example.

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastToParam>)
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr<mxnet::FInferShape>("FInferShape", BroadcastToShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

// backward op for broadcast.
NNVM_REGISTER_OP(_broadcast_backward)
.set_attr_parser(ParamParser<ReduceAxesParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ReduceAxesCompute<cpu, mshadow::red::sum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

NNVM_REGISTER_OP(broadcast_like)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
      return std::vector<std::string>{"lhs", "rhs"};
    })
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,
    const std::vector<nnvm::NodeEntry>& ograds) {
      if (CheckGradAllZero(ograds))
        return MakeZeroGradNodes(n, ograds);
      std::vector<nnvm::NodeEntry> lhs = MakeNonlossGradNode("_broadcast_backward", n, ograds, {},
            {{"keepdims", "true"}});
      lhs.emplace_back(MakeNode("zeros_like", n->attrs.name + "_rhs_backward",
                       {n->inputs[1]}, nullptr, &n));
      return lhs;
    })
.add_argument("lhs", "NDArray-or-Symbol", "First input.")
.add_argument("rhs", "NDArray-or-Symbol", "Second input.")
.describe(R"code(Broadcasts lhs to have the same shape as rhs.

Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

   broadcast_like([[1,2,3]], [[5,6,7],[7,8,9]]) = [[ 1.,  2.,  3.],
                                                   [ 1.,  2.,  3.]])

   broadcast_like([9], [1,2,3,4,5], lhs_axes=(0,), rhs_axes=(-1,)) = [9,9,9,9,9]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastLikeParam>)
.add_arguments(BroadcastLikeParam::__FIELDS__())
.set_attr<mxnet::FInferShape>("FInferShape", BroadcastLikeShape)
.set_attr<FCompute>("FCompute<cpu>", BroadcastCompute<cpu>);

}  // namespace op
}  // namespace mxnet
