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
 * Copyright (c) 2016 by Contributors
 * \file graph_algorithm.h
 * \brief This header contains graph algorithms on StaticGraph.
 *  It is used  compute informations such as whether two
 *  operations can run in parallel, and helps allocation.
*/
#ifndef MXNET_NNVM_GRAPH_ALGORITHM_H_
#define MXNET_NNVM_GRAPH_ALGORITHM_H_

#include <nnvm/graph.h>
#include <vector>

namespace nnvm {
namespace pass {

/*!
 * \brief Find best path in the DAG, with reward defined
 *  by sum of reward of each node along the path.
 * \param graph the original static graph.
 * \param topo_order topo order of the nodes in the graph.
 * \param node_reward the reward of each node.
 * \param path the output path of nodes.
 * \return the total reward of best path.
 */
inline uint32_t MXFindBestPath(
    const IndexedGraph& graph,
    const std::vector<uint32_t>& node_reward,
    std::vector<uint32_t>* path) {
  const uint32_t num_nodes = static_cast<uint32_t>(graph.num_nodes());
  CHECK_EQ(num_nodes, node_reward.size());

  std::vector<uint32_t> best_reward(node_reward.size(), 0);
  std::vector<uint32_t> next_node(node_reward.size(), num_nodes);
  uint32_t best_solution = 0, best_start_node = 0;

  // traverse in reverse topo order
  for (uint32_t i = static_cast<uint32_t>(graph.num_nodes()); i != 0; --i) {
    const uint32_t nid = i - 1;
    best_reward[nid] += node_reward[nid];
    if (best_reward[nid] > best_solution) {
      best_solution = best_reward[nid];
      best_start_node = nid;
    }
    for (const auto& e : graph[nid].inputs) {
      const uint32_t prev = e.node_id;
      if (best_reward[nid] > best_reward[prev]) {
        best_reward[prev] = best_reward[nid];
        next_node[prev] = nid;
      }
    }
  }
  path->clear();
  uint32_t reward = 0;
  for (uint32_t nid = best_start_node; nid < num_nodes; nid = next_node[nid]) {
    path->push_back(nid); reward += node_reward[nid];
  }
  CHECK_EQ(reward, best_solution);
  return best_solution;
}

/*!
 * \brief Color the nodes in the graph into index.
 *  The coloring algorithm tries to assign node group
 *  such that node in the same group cannot run in parallel.
 *
 * \param graph the original indexed graph.
 * \param node_importance The importance of the node
 * \param max_ncolor maximum number of colors allowed.
 * \param color the color index of each of the node.
 * \return the total number of colors.
 */
inline uint32_t MXColorNodeGroup(
    const IndexedGraph &graph,
    std::vector<uint32_t> node_importance,
    uint32_t max_ncolor,
    std::vector<uint32_t> *color) {
  CHECK_NE(max_ncolor, 0U);
  CHECK_EQ(graph.num_nodes(), node_importance.size());

  color->clear();
  color->resize(graph.num_nodes(), max_ncolor);
  uint32_t cindex;
  // greedy algorithm, every time
  // find a path with best reward and assign a new color
  // All the nodes in the path cannot run in parallel.
  for (cindex = 0; cindex < max_ncolor - 1; ++cindex) {
    std::vector<uint32_t> path;
    uint32_t reward = MXFindBestPath(graph, node_importance, &path);
    if (reward == 0) break;
    for (uint32_t nid : path) {
      if (node_importance[nid] != 0) {
        CHECK_EQ(color->at(nid), max_ncolor);
        color->at(nid) = cindex;
        // make the importance 0 after color is decided.
        node_importance[nid] = 0;
      }
    }
  }
  // assign i for rest of the node
  for (uint32_t i = 0; i < graph.num_nodes(); ++i) {
    if (color->at(i) == max_ncolor) {
      color->at(i) = cindex;
    }
  }
  return cindex + 1;
}

}  // namespace pass
}  // namespace nnvm

#endif  // MXNET_NNVM_GRAPH_ALGORITHM_H_
