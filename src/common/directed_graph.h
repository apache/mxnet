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
 * \file static_graph.h
 * \author Pedro Larroy
 * \brief directed static graph
 */

#include <vector>
#include <map>
#include <algorithm>
#include <functional>
namespace common {
namespace graph {

struct Empty {};

template<class Node_=Empty, class EdgeAttrs_=Empty, typename NodeKey_t_=size_t, typename EdgeKey_t_=size_t>
class DirectedGraph {
 protected:
  const EdgeAttrs_ default_edge = EdgeAttrs_();
 public:
  typedef Node_ Node;
  typedef NodeKey_t_ NodeKey_t;
  typedef EdgeKey_t_ EdgeKey_t;
  typedef EdgeAttrs_ EdgeAttrs;

  struct Edge : EdgeAttrs {
    Edge(NodeKey_t src, NodeKey_t dst, const EdgeAttrs& attrs) :
        EdgeAttrs(attrs),
        src(src),
        dst(dst)
     {
     }
    NodeKey_t src;
    NodeKey_t dst;
    bool operator<(const Edge &other) const {
      if (src != other.src)
        return src < other.src;
      return dst < other.dst;
    }
    bool operator==(const Edge &other) const {
      if (src != other.src)
        return false;
      return dst != other.dst;
    }
  };

  DirectedGraph(NodeKey_t num_nodes = 0) :
      edges() {
    nodes.reserve(num_nodes);
  }

  void addNode(const Node& node_attrs) {
    nodes.emplace_back(node_attrs);
  }

  Node& node(NodeKey_t node_key) {
    return nodes.at(node_key);
  }

  EdgeKey_t numEdges() const {
    return static_cast<EdgeKey_t>(edges.size());
  }

  NodeKey_t numNodes() const {
    return static_cast<NodeKey_t>(nodes.size());
  }

  void addEdge(NodeKey_t src, NodeKey_t dst, const EdgeAttrs& edge_attrs) {
    const auto reqsz = static_cast<size_t>(std::max(src, dst)) + 1;
    if (reqsz > nodes.size())
      nodes.resize(reqsz);
    edges.emplace(src, Edge{src, dst, edge_attrs});
  }
  void addEdge(NodeKey_t src, NodeKey_t dst) {
    addEdge(src, dst, default_edge);
  }

  typedef typename std::vector<Node>::const_iterator NodeIterator;
  typedef typename std::multimap<NodeKey_t, Edge>::const_iterator EdgeIterator;

  EdgeIterator edgesBegin() const {
    return edges.begin();
  }

  EdgeIterator edgesEnd() const {
    return edges.end();
  }

  EdgeIterator outEdgesBegin(NodeKey_t node) const {
    return edges.lower_bound(node);
  }

  EdgeIterator outEdgesEnd(NodeKey_t node) const {
    return edges.upper_bound(node);
  }

  NodeIterator nodesBegin() const {
    return nodes.begin();
  }

  NodeIterator nodesEnd() const {
    return nodes.end();
  }

 protected:
  std::vector<Node> nodes;
  std::multimap<NodeKey_t, Edge> edges;
};

};  // end namespace common
};  // end namespace graph
