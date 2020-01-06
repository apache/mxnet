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
bool CutGraphInputs(const std::vector<nnvm::NodeEntry *> &input_entries,
                    bool skip_var, std::vector<nnvm::NodeEntry> *orig_entries) {
  struct pred_entry {
    nnvm::NodeEntry e;
    explicit pred_entry(const nnvm::NodeEntry &_e): e(_e) {}
    bool operator()(const nnvm::NodeEntry e1) {
      return e.node == e1.node && e.index == e1.index;
    }
  };

  std::vector<nnvm::NodePtr> var_nodes;
  orig_entries->clear();
  orig_entries->reserve(input_entries.size());
  for (auto input_entry : input_entries) {
    // If the node is a variable itself, we may want to skip the node.
    if (input_entry->node->is_variable() && skip_var)
      continue;

    auto it = std::find_if(orig_entries->begin(), orig_entries->end(),
                           pred_entry(*input_entry));
    bool exist = (it != orig_entries->end());
    orig_entries->push_back(*input_entry);
    nnvm::NodePtr n;
    // If we haven't seen the entry before, we need to create a new var node
    // for the node entry.
    if (!exist) {
      nnvm::Symbol sym;
      sym.outputs.push_back(*input_entry);
      n = nnvm::CreateVariableNode(sym.ListOutputNames()[0]);
    } else {
      // Otherwise, we use the var node created before.
      size_t idx = it - orig_entries->begin();
      CHECK_LT(idx, var_nodes.size());
      n = var_nodes[idx];
    }
    var_nodes.push_back(n);
    *input_entry = nnvm::NodeEntry{n, 0, 0};
  }
  return true;
}

}  // namespace mxnet
