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
#include "common/directed_graph.h"
#include "dmlc/json.h"
#include <vector>
#include <string>

using std::vector;
using std::string;
using namespace std;
using common::graph::DirectedGraph;

using namespace nnvm;

namespace {
  struct DumpNode {
    NodePtr node_ptr_;
    bool is_input;
    bool is_op;
    bool is_output;
  };
  typedef DirectedGraph<DumpNode> DumpGraph_t;

  /**
   * Serialize a graph to Json
   * @param g
   * @return json content
   */
  std::string SerializeJson(DumpGraph_t& g) {
    ostringstream os;
    dmlc::JSONWriter json_writer(&os);
    json_writer.BeginObject();
    vector<string> inputs;
    vector<string> outputs;
    for (const DumpNode& node : g) {
      if (node.is_input)
        inputs.push_back(node.node_ptr_->attrs.name);
      else if (node.is_output)
        outputs.push_back(node.node_ptr_->attrs.name);
    }
    json_writer.WriteObjectKeyValue("inputs", inputs);
    json_writer.WriteObjectKeyValue("outputs", outputs);
    json_writer.EndObject();
    return os.str();
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
  std::string SerializeDot(DumpGraph_t& g) {
    ostringstream os;
    os << "digraph G {" << endl;
    for (auto edge_it : g.edges()) {
      string src_name = g.node(edge_it->src).node_ptr_->attrs.name;
      string dst_name = g.node(edge_it->dst).node_ptr_->attrs.name;
      os << "  " << src_name << " -> " << dst_name << endl;
    }
    os << "}";
    return os.str();
  }
}  // end anon ns

namespace nnvm {


std::string GraphDump(const Graph& graph) {
    vector<NodePtr> topo_order;
    DFSVisit(graph.outputs, [&](const NodePtr& nodePtr) {
        cout << "Node: " << nodePtr.get() << " " << nodePtr->attrs.name << endl;
        topo_order.push_back(nodePtr);
    });
    set<NodePtr> outputs;
    transform(begin(graph.outputs), end(graph.outputs), inserter(outputs, end(outputs)),
        [](const NodeEntry& ne) { return ne.node; });

    // Build directed graph;
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

    for (const DumpNode& node : g) {
      node_key_t dst_key = node_ptr_to_node_key.at(node.node_ptr_);
      for (const NodeEntry& node_entry: node.node_ptr_->inputs) {
        node_key_t src_key = node_ptr_to_node_key.at(node_entry.node);
        g.addEdge(src_key, dst_key);
      }
    }
    return SerializeDot(g);
}


}  // end namespace nnvm
