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
#ifndef MXNET_COMMON_DIRECTED_GRAPH_H_
#define MXNET_COMMON_DIRECTED_GRAPH_H_

/*!
 * \file static_graph.h
 * \author Pedro Larroy
 * \brief directed static graph
 */

#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <iterator>

#define SELF_t \
    static auto helper() -> typename std::remove_reference<decltype(*this)>::type; \
    typedef decltype(helper())

namespace common {
namespace graph {

struct Empty {};

template<class Node_ = Empty, class EdgeAttrs_ = Empty,
    typename NodeKey_t_ = size_t, typename EdgeKey_t_ = size_t>
class DirectedGraph {
 protected:
  const EdgeAttrs_ default_edge = EdgeAttrs_();

 public:
  SELF_t DirectedGraph_t;
  typedef Node_ Node;
  typedef NodeKey_t_ NodeKey_t;
  typedef EdgeKey_t_ EdgeKey_t;
  typedef EdgeAttrs_ EdgeAttrs;

  struct Edge : EdgeAttrs {
    Edge(NodeKey_t src, NodeKey_t dst, const EdgeAttrs& attrs) :
        EdgeAttrs(attrs),
        src(src),
        dst(dst) {
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

  typedef typename std::vector<Node>::const_iterator NodeIterator;
  typedef typename std::multimap<NodeKey_t, Edge>::const_iterator InternalEdgeIterator;


  struct EdgeIterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Edge;
    using reference_type = Edge&;
    using const_pointer_type = const Edge*;
    InternalEdgeIterator internal_edge_iterator_;
    explicit EdgeIterator(const InternalEdgeIterator& ei) :
      internal_edge_iterator_(ei) {
    }
    EdgeIterator& operator++() {
      ++internal_edge_iterator_;
      return *this;
    }
    const_pointer_type operator*() const {
      return static_cast<const Edge*>(&(internal_edge_iterator_->second));
    }
    const_pointer_type operator->() const {
      return &(internal_edge_iterator_->second);
    }
    bool operator==(const EdgeIterator& other) const {
      if (internal_edge_iterator_ != other.internal_edge_iterator_)
        return false;
      return true;
    }
    bool operator!=(const EdgeIterator& other) const {
      return !((*this) == other);
    }
  };

  struct EdgeRange {
    const DirectedGraph_t& graph_;
    explicit EdgeRange(const DirectedGraph_t& g) :
      graph_(g) {
    }
    EdgeIterator begin() {
      return EdgeIterator(std::begin(graph_.edges_));
    }
    EdgeIterator end() {
      return EdgeIterator(std::end(graph_.edges_));
    }
  };

  explicit DirectedGraph(NodeKey_t num_nodes = 0) :
      edges_() {
    nodes_.reserve(num_nodes);
  }

  NodeKey_t addNode(const Node& node_attrs) {
    nodes_.emplace_back(node_attrs);
    return nodes_.size() - 1;
  }

  template <typename... Args>
  NodeKey_t addNode(Args&&... args) {
    nodes_.emplace_back(std::forward<Args>(args)...);
    return nodes_.size() - 1;
  }

  Node& node(NodeKey_t node_key) {
    return nodes_.at(node_key);
  }

  EdgeRange edges() const {
    return EdgeRange(*this);
  }

  EdgeKey_t numEdges() const {
    return static_cast<EdgeKey_t>(edges_.size());
  }

  NodeKey_t numNodes() const {
    return static_cast<NodeKey_t>(nodes_.size());
  }

  void addEdge(NodeKey_t src, NodeKey_t dst, const EdgeAttrs& edge_attrs) {
    const auto reqsz = static_cast<size_t>(std::max(src, dst)) + 1;
    if (reqsz > nodes_.size())
      nodes_.resize(reqsz);
    edges_.emplace(src, Edge{src, dst, edge_attrs});
  }

  void addEdge(NodeKey_t src, NodeKey_t dst) {
    addEdge(src, dst, default_edge);
  }


  EdgeIterator edgesBegin() const {
    return EdgeIterator(std::begin(edges_));
  }

  EdgeIterator edgesEnd() const {
    return EdgeIterator(std::end(edges_));
  }

  EdgeIterator outEdgesBegin(NodeKey_t node) const {
    return EdgeIterator(edges_.lower_bound(node));
  }

  EdgeIterator outEdgesEnd(NodeKey_t node) const {
    return EdgeIterator(edges_.upper_bound(node));
  }

  NodeIterator nodesBegin() const {
    return nodes_.begin();
  }

  NodeIterator nodesEnd() const {
    return nodes_.end();
  }

  NodeIterator begin() const {
    return nodesBegin();
  }

  NodeIterator end() const {
    return nodesEnd();
  }

 protected:
  std::vector<Node> nodes_;
  std::multimap<NodeKey_t, Edge> edges_;
};

};  // namespace graph
};  // namespace common
#endif  // MXNET_COMMON_DIRECTED_GRAPH_H_
