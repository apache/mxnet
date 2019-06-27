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

#include <gtest/gtest.h>
#include "common/directed_graph.h"

using namespace common::graph;
using namespace std;

TEST(DirectedGraphTest, basic) {
  DirectedGraph<> g;
  g.addEdge(0, 4);
  g.addEdge(3, 5);
  g.addEdge(2, 1);
  EXPECT_EQ(g.numNodes(), 6);
  EXPECT_EQ(g.numEdges(), 3);
}

namespace {

struct Node {
  string opName;
  size_t num_inputs;
  bool operator==(const Node& o) const {
    if (opName != o.opName)
      return false;
    if (num_inputs != o.num_inputs)
      return false;
    return true;
  }
};

struct Edge {
  Edge() : cost(1) {}
  explicit Edge(size_t cost) : cost(cost) {}
  size_t cost;
};

};  // namespace

TEST(DirectedGraphTest, withPayload) {
  DirectedGraph<Node, Edge> g;
  auto x = Node{"x", 0};
  auto w = Node{"w", 0};
  auto b = Node{"b", 0};
  g.addNode(x);  // 0
  g.addNode(w);  // 1
  g.addNode(b);  // 2
  g.addNode(Node{"mul", 2});  // 3
  g.addNode(Node{"add", 2});  // 4
  g.addEdge(0, 3);
  g.addEdge(1, 3);
  auto mul_edge_rhs = Edge{10};
  auto mul_edge_lhs = Edge{10};
  g.addEdge(2, 4, mul_edge_rhs);
  g.addEdge(3, 4, mul_edge_lhs);

  EXPECT_EQ(g.numNodes(), 5);
  EXPECT_EQ(g.numEdges(), 4);
  EXPECT_EQ(g.node(0), x);
  EXPECT_EQ(g.node(1), w);
  EXPECT_EQ(g.node(2), b);
  EXPECT_EQ(g.node(3), (Node{"mul", 2}));
  EXPECT_EQ(g.node(4), (Node{"add", 2}));
}

TEST(DirectedGraphTest, iterators) {
  DirectedGraph<Node, Edge> g;
  auto x = Node{"x", 0};
  auto w = Node{"w", 0};
  auto b = Node{"b", 0};
  g.addNode(x);  // 0
  g.addNode(w);  // 1
  g.addNode(b);  // 2
  g.addNode(Node{"mul", 2});  // 3
  g.addNode(Node{"add", 2});  // 4
  g.addEdge(0, 3);  // 0
  g.addEdge(1, 3);  // 1
  auto mul_edge_rhs = Edge{10};
  auto mul_edge_lhs = Edge{10};
  g.addEdge(2, 4, mul_edge_rhs);  // 2
  g.addEdge(3, 4, mul_edge_lhs);  // 3
  size_t edge_cnt = 0;
  vector<size_t> costs;
  for (auto ei = g.edgesBegin(); ei != g.edgesEnd(); ++ei) {
    costs.push_back(ei->cost);
    ++edge_cnt;
  }
  EXPECT_EQ(edge_cnt, g.numEdges());
  EXPECT_EQ(costs.at(0), 1);
  EXPECT_EQ(costs.at(1), 1);
  EXPECT_EQ(costs.at(2), 10);
  EXPECT_EQ(costs.at(3), 10);
  size_t node_cnt = 0;
  for (auto ei = g.nodesBegin(); ei != g.nodesEnd(); ++ei)
    ++node_cnt;
  EXPECT_EQ(node_cnt, 5);
}

TEST(DirectedGraphTest, iterators2) {
  DirectedGraph<> g;
  g.addEdge(3, 5);
  g.addEdge(0, 1);
  g.addEdge(0, 1);
  g.addEdge(2, 4);
  g.addEdge(0, 1);
  g.addEdge(2, 5);

  typedef DirectedGraph<>::NodeKey_t node_t;
  vector<pair<node_t, node_t>> edges;
  for (auto ei = g.outEdgesBegin(0); ei != g.outEdgesEnd(0); ++ei)
    edges.emplace_back(ei->src, ei->dst);
  EXPECT_EQ(edges.size(), 3);
  EXPECT_EQ(edges.at(0), make_pair(node_t(0), node_t(1)));
  EXPECT_EQ(edges.at(1), make_pair(node_t(0), node_t(1)));
  EXPECT_EQ(edges.at(2), make_pair(node_t(0), node_t(1)));

  edges.clear();
  for (auto ei = g.outEdgesBegin(1); ei != g.outEdgesEnd(1); ++ei)
    edges.emplace_back(ei->src, ei->dst);
  EXPECT_EQ(edges.size(), 0);

  edges.clear();
  for (auto ei = g.outEdgesBegin(2); ei != g.outEdgesEnd(2); ++ei)
    edges.emplace_back(ei->src, ei->dst);
  EXPECT_EQ(edges.size(), 2);
  EXPECT_EQ(edges.at(0), make_pair(node_t(2), node_t(4)));
  EXPECT_EQ(edges.at(1), make_pair(node_t(2), node_t(5)));

  edges.clear();
  for (auto ei = g.outEdgesBegin(3); ei != g.outEdgesEnd(3); ++ei)
    edges.emplace_back(ei->src, ei->dst);
  EXPECT_EQ(edges.size(), 1);

  edges.clear();
  for (auto ei = g.outEdgesBegin(8); ei != g.outEdgesEnd(8); ++ei)
    edges.emplace_back(ei->src, ei->dst);
  EXPECT_EQ(edges.size(), 0);

  edges.clear();
  for (auto ei : g.edges()) {
    edges.emplace_back(ei->src, ei->dst);
  }
  EXPECT_EQ(edges.size(), 6);
  typedef pair<size_t, size_t> p_t;
  EXPECT_TRUE(edges[0] == p_t(0, 1));
  EXPECT_TRUE(edges[1] == p_t(0, 1));
  EXPECT_TRUE(edges[2] == p_t(0, 1));
  EXPECT_TRUE(edges[3] == p_t(2, 4));
  EXPECT_TRUE(edges[4] == p_t(2, 5));
  EXPECT_TRUE(edges[5] == p_t(3, 5));
}

