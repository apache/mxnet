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
 * \file control_flow_op.cc
 * \brief CPU Implementation of flow control
 */
#include "./control_flow_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(where)
MXNET_ADD_SPARSE_OP_ALIAS(where)
.describe(R"code(Return the elements, either from x or y, depending on the condition.

Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,
depending on the elements from condition are true or false. x and y must have the same shape.
If condition has the same shape as x, each element in the output array is from x if the
corresponding element in the condition is true, and from y if false.

If condition does not have the same shape as x, it must be a 1D array whose size is
the same as x's first dimension size. Each row of the output array is from x's row
if the corresponding element from condition is true, and from y's row if false.

Note that all non-zero values are interpreted as ``True`` in condition.

Examples::

  x = [[1, 2], [3, 4]]
  y = [[5, 6], [7, 8]]
  cond = [[0, 1], [-1, 0]]

  where(cond, x, y) = [[5, 2], [3, 8]]

  csr_cond = cast_storage(cond, 'csr')

  where(csr_cond, x, y) = [[5, 2], [3, 8]]

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"condition", "x", "y"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", WhereOpShape)
.set_attr<nnvm::FInferType>("FInferType", WhereOpType)
.set_attr<FInferStorageType>("FInferStorageType", WhereOpForwardStorageType)
.set_attr<FCompute>("FCompute<cpu>", WhereOpForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", WhereOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  // Use the following lambda function instead of ElemwiseGradUseIn
  // for best efficiency. grad[condition] = 0; to calculate grad[x] and grad[y]
  // we need only condition from input.
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> ret;
    // make zero grad node for grad[condition]
    auto p = MakeNode("zeros_like", n->attrs.name + "_cond_backward",
                      {n->inputs[0]}, nullptr, &n);
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});

    // make grad nodes for grad[x] and grad[y]
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[0]);  // only need condition to calculate gradients
    p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("_backward_where");
    p->attrs.name = n->attrs.name + "_backward";
    p->attrs.dict = n->attrs.dict;
    if (p->op()->attr_parser != nullptr) {
      p->op()->attr_parser(&(p->attrs));
    }
    p->control_deps.emplace_back(n);
    p->inputs = std::move(heads);
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p, 1, 0});

    return ret;
  })
.add_argument("condition", "NDArray-or-Symbol", "condition array")
.add_argument("x", "NDArray-or-Symbol", "")
.add_argument("y", "NDArray-or-Symbol", "");

NNVM_REGISTER_OP(_backward_where)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", WhereOpBackwardStorageType)
.set_attr<FCompute>("FCompute<cpu>", WhereOpBackward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", WhereOpBackwardEx<cpu>);


}  // namespace op
}  // namespace mxnet
