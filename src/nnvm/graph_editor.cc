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
#include <nnvm/node.h>

namespace nnvm {
NodePtr CreateVariableNode(const std::string& name);
}

namespace mxnet {

/*
 * Given a computation graph, this function finds the input nodes of the graph
 * and create symbols for the input nodes. It returns the input symbols.
 */
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

/*
 * Given a computation graph and a set of input node entries, this function cuts
 * the node entries and creates new variable nodes as the input nodes of the
 * subgraph. It returns the nodes that connect to the subgraph directly and
 * the names of the new variable nodes.
 */
bool CutGraph(const std::vector<nnvm::NodeEntry *> &input_entries,
              const std::string &in_name_prefix, bool skip_var,
              std::vector<nnvm::NodeEntry> *orig_entries,
              std::vector<std::string> *new_var_names) {
  orig_entries->reserve(input_entries.size());
  for (size_t i = 0; i < input_entries.size(); i++) {
    nnvm::NodeEntry *e = input_entries[i];
    // If the node is a variable itself, we may want to skip the node.
    if (e->node->is_variable() && skip_var)
      continue;

    orig_entries->push_back(*e);
    new_var_names->push_back(in_name_prefix + std::to_string(i));
    nnvm::NodePtr n = nnvm::CreateVariableNode(new_var_names->back());
    *e = nnvm::NodeEntry{n, 0, 0};
  }
  return true;
}

}
