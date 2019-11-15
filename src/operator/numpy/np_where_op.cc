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
 * \file np_where_op.cc
 * \brief CPU Implementation of numpy operator where
 */

#include "np_where_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_where)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"condition", "x", "y"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyWhereOpShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyWhereOpType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{1, 0}, {2, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyWhereOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  // Use the following lambda function instead of ElemwiseGradUseIn
  // for best efficiency. grad[condition] = 0; to calculate grad[x] and grad[y]
  // we need only condition from input.
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> ret;
    // make zero grad node for grad[condition]
    auto p = MakeNode("zeros_like", n->attrs.name + "_cond_backward",
                      {n->inputs[0]}, nullptr, &n);
    ret.emplace_back(p);

    // make grad nodes for grad[x] and grad[y]
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[0]);  // only need condition to calculate gradients
    p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("_backward_np_where");
    p->attrs.name = n->attrs.name + "_backward";
    p->attrs.dict = n->attrs.dict;
    if (p->op()->attr_parser != nullptr) {
      p->op()->attr_parser(&(p->attrs));
    }
    p->control_deps.emplace_back(n);
    p->inputs = std::move(heads);
    ret.emplace_back(p, 0, 0);
    ret.emplace_back(p, 1, 0);
    return ret;
  })
.add_argument("condition", "NDArray-or-Symbol", "condition array")
.add_argument("x", "NDArray-or-Symbol", "input x")
.add_argument("y", "NDArray-or-Symbol", "input y");

NNVM_REGISTER_OP(_backward_np_where)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyWhereOpBackward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

}  // namespace op
}  // namespace mxnet
