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
 * \file graph_dump.cc
 * \brief Utilities to introspect and print the execution graph
 * \author Pedro Larroy
 */

#include "./graph_dump.h"
#include <vector>
#include <string>
#include "../common/directed_graph.h"
#include "dmlc/json.h"
#include "nnvm/graph.h"

using std::vector;
using std::string;
using namespace std;
using common::graph::DirectedGraph;

using namespace nnvm;

namespace {
// Helper class
struct DumpNode {
  NodePtr node_ptr_;
  bool is_input;
  bool is_op;
  bool is_output;
};
typedef DirectedGraph<DumpNode> DumpGraph_t;

std::string unnamed(const std::string &s) {
  if (s.empty())
    return "unnamed";
  return s;
}

/**
 * Serialize a graph to dot format
 * @param g a graph to serialize to dot format
 * @return a string with a dot file content
 * example result:
 * digraph G {
 *      x -> x_mul_w
 *      w -> x_mul_w
 * }
 */
std::string SerializeDot(const DumpGraph_t& g) {
  ostringstream os;
  os << "digraph G {" << endl;
  for (auto edge_it : g.edges()) {
    ostringstream src_name_os;
    ostringstream dst_name_os;
    const NodePtr& src = g.node(edge_it->src).node_ptr_;
    const NodePtr& dst = g.node(edge_it->dst).node_ptr_;
    if (src->op())
      src_name_os << src->op()->name << " ";
    if (dst->op())
      dst_name_os << dst->op()->name << " ";
    src_name_os << unnamed(src->attrs.name);
    dst_name_os << unnamed(dst->attrs.name);
    os << "  \"" << src_name_os.str() << "\" -> \"" << dst_name_os.str() << "\"" << endl;
  }
  os << "}";
  return os.str();
}

}  // namespace

namespace nnvm {


/**
 * We convert the NNVM graph to a directed graph to enumerate the edges and call a serialization
 * function
 */
std::string GraphDump(const std::vector<NodeEntry>& output_nodes) {
    // Traverse the NNVM graph in topological order
    vector<NodePtr> topo_order;
    DFSVisit(output_nodes, [&](const NodePtr& nodePtr) {
        topo_order.push_back(nodePtr);
    });
    set<NodePtr> outputs;
    transform(begin(output_nodes), end(output_nodes), inserter(outputs, end(outputs)),
        [](const NodeEntry& ne) { return ne.node; });

    // Build a generic directed graph, gather nodes
    DumpGraph_t g;
    typedef DumpGraph_t::NodeKey_t node_key_t;
    unordered_map<NodePtr, node_key_t > node_ptr_to_node_key;
    for (const NodePtr& node_ptr : topo_order) {
      DumpNode node{node_ptr,
                node_ptr->is_variable(),
                node_ptr->op() != nullptr,
                outputs.count(node_ptr) != 0};
      DumpGraph_t::NodeKey_t node_key = g.addNode(move(node));
      node_ptr_to_node_key.emplace(node_ptr, node_key);
    }

    // Add edges to the graph
    for (const DumpNode& node : g) {
      node_key_t dst_key = node_ptr_to_node_key.at(node.node_ptr_);
      // Use inputs from the nodes to get edges
      for (const NodeEntry& node_entry : node.node_ptr_->inputs) {
        node_key_t src_key = node_ptr_to_node_key.at(node_entry.node);
        g.addEdge(src_key, dst_key);
      }
    }
    return SerializeDot(g);
}


}  // end namespace nnvm
