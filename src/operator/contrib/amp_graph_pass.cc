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
 *  Copyright (c) 2019 by Contributors
 * \file amp_graph_pass.cc
 * \brief graph pass regarding AMP
 * \author Clement Fuji Tsang
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>

namespace mxnet {
namespace op {

using nnvm::Node;
using nnvm::NodePtr;
using nnvm::Graph;


/*
 * \brief Remove amp_cast and amp_multicast and replug the fp32 weights
 */
Graph RemoveAmpCast(Graph&& g) {
  DFSVisit(g.outputs, [](const NodePtr& n) {
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      auto e = n->inputs[i];
      if (e.node->op() == Op::Get("amp_cast")) {
        n->inputs[i] = e.node->inputs[0];
      } else if (e.node->op() == Op::Get("amp_multicast")) {
        n->inputs[i] = e.node->inputs[e.index];
      }
    }
  });
  return g;
}

NNVM_REGISTER_PASS(RemoveAmpCast)
.describe("")
.set_body(RemoveAmpCast)
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
