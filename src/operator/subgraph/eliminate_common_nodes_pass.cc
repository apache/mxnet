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
 * \file eliminate_common_nodes_pass.cc
 * \brief Graph pass to eliminate common nodes from the input graph
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>

#include "imperative/exec_pass.h"

namespace mxnet {

nnvm::Graph EliminateCommonNodesPass(nnvm::Graph&& g) {
  const int enabled = dmlc::GetEnv("MXNET_NODE_ELIMINATION", 1);
  if (enabled == 0) {
    LOG(INFO) << "Skipping common nodes elimination.";
    return std::move(g);
  }

  return exec::EliminateCommonExpr(std::move(g));
}

NNVM_REGISTER_PASS(EliminateCommonNodesPass)
    .describe("Removes additional Nodes with identical inputs and function.")
    .set_body(EliminateCommonNodesPass)
    .set_change_graph(true);

}  // namespace mxnet
