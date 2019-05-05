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
 * Copyright (c) 2019 by Contributors
 * \file node_entry_count.cc
 * \brief function that count how many times a node entry is used
 * \author Clement Fuji Tsang
 */
#include "./exec_pass.h"

// TODO(cfujitsang): should this be pushed to nnvm repository ?
namespace mxnet {
namespace exec {

NodeEntryMapCounter GetNodeEntryCount(const nnvm::Graph& g) {
  NodeEntryMapCounter outputs;
  DFSVisit(g.outputs, [&outputs](const nnvm::NodePtr& node) {
    for (auto e : node->inputs) {
      outputs[e]++;
    }
  });
  for (auto e : g.outputs) {
    outputs[e]++;
  }
  return outputs;
}

}  // namespace exec
}  // namespace mxnet
