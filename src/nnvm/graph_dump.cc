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
#include <vector>
#include <string>

using std::vector;
using std::string;
using namespace std;
using common::graph::DirectedGraph;

using namespace nnvm;

namespace {
  struct Node {
    NodePtr node_ptr_;
  };
}  // end anon ns

namespace nnvm {


std::string GraphDump(const Graph& graph) {
    vector<NodePtr> topo_order;
    DFSVisit(graph.outputs, [&](const NodePtr& nodePtr) {
        //cout << "Node: " << nodePtr.get() << " " << nodePtr->attrs.name << endl;
        topo_order.push_back(nodePtr);
    });
    DirectedGraph<NodePtr> directed_graph;
    for (const auto node_ptr : topo_order) {
      directed_graph.addNode(node_ptr);
    }
    return "";
}

}  // end namespace nnvm
