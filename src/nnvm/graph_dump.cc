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
#include "nnvm/graph.h"

using std::vector;
using std::string;
using namespace std;

using namespace nnvm;

namespace {

std::string unnamed(const std::string &s) {
  if (s.empty())
    return "unnamed";
  return s;
}

}  // namespace

namespace nnvm {


/**
 * We convert the NNVM graph to a directed graph to enumerate the edges and call a serialization
 * function
 */
std::string GraphDump(const std::vector<NodeEntry>& output_nodes) {
    ostringstream os;
    os << "digraph G {" << endl;
    Graph g;
    g.outputs = output_nodes;
    auto& indexed_graph = g.indexed_graph();
    for (size_t i=0; i<indexed_graph.num_nodes(); ++i) {
      const IndexedGraph::Node& idst= indexed_graph[i];
      for (const IndexedGraph::NodeEntry& input : idst.inputs) {
        const Node& dst = (*idst.source);
        const IndexedGraph::Node& isrc = indexed_graph[input.node_id];
        const Node& src = (*isrc.source);
        ostringstream src_name_os;
        ostringstream dst_name_os;
        if (src.op())
          src_name_os << src.op()->name << " ";
        if (dst.op())
          dst_name_os << dst.op()->name << " ";
        src_name_os << unnamed(src.attrs.name);
        dst_name_os << unnamed(dst.attrs.name);
        os << "  \"" << src_name_os.str() << "\" -> \"" << dst_name_os.str() << "\"" << endl;
      }
    }
    os << "}";
    return os.str();
}


}  // end namespace nnvm
