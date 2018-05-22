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
 * \file graph_editor.cc
 * The functions in this file edit an NNVM graph. Potentially,
 * these functions should be moved to NNVM in the future.
 */

#include <nnvm/symbolic.h>
#include <nnvm/graph.h>

namespace mxnet {

std::vector<nnvm::Symbol *> GetInputSymbols(const nnvm::Symbol &sym) {
  nnvm::Graph g;
  std::vector<nnvm::Symbol *> input_syms;
  g.outputs = sym.outputs;
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  // Go through all nodes and return the ones representing variables.
  for (size_t i = 0; i < idx.num_nodes(); i++) {
    const nnvm::Node &n = *idx[i].source;
    for (const nnvm::NodeEntry &e : n.inputs) {
      auto p = e.node;
      if (p->is_variable()) {
        nnvm::Symbol *s = new nnvm::Symbol();
        s->outputs.push_back(e);
        input_syms.push_back(s);
      }
    }
  }
  return input_syms;
}

}
