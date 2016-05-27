/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_allocation_helper.h
 * \brief This header contains graph algorithms on StaticGraph.
 *  It is used  compute informations such as whether two
 *  operations can run in parallel, and helps allocation.
*/
#ifndef MXNET_SYMBOL_GRAPH_ALGORITHM_H_
#define MXNET_SYMBOL_GRAPH_ALGORITHM_H_

#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <utility>
#include "./static_graph.h"

namespace mxnet {
namespace graph {
/*!
 * \brief Find best path in the DAG, with reward defined
 *  by sum of reward of each node along the path.
 * \param graph the original static graph.
 * \param topo_order topo order of the nodes in the graph.
 * \param node_reward the reward of each node.
 * \param path the output path of nodes.
 * \return the total reward of best path.
 */
inline uint32_t FindBestPath(
    const StaticGraph &graph,
    const std::vector<uint32_t> &topo_order,
    const std::vector<uint32_t> &node_reward,
    std::vector<uint32_t> *path) {
  const uint32_t num_nodes = static_cast<uint32_t>(graph.nodes.size());
  CHECK_EQ(graph.nodes.size(), node_reward.size());
  CHECK_EQ(graph.nodes.size(), topo_order.size());

  std::vector<uint32_t> best_reward(node_reward.size(), 0);
  std::vector<uint32_t> next_node(node_reward.size(), num_nodes);
  uint32_t best_solution = 0, best_start_node = 0;

  // traverse in reverse topo order
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    const uint32_t nid = *it;
    best_reward[nid] += node_reward[nid];
    if (best_reward[nid] > best_solution) {
      best_solution = best_reward[nid];
      best_start_node = nid;
    }
    for (const StaticGraph::DataEntry& e : graph.nodes[nid].inputs) {
      const uint32_t prev = e.source_id;
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
 * \param graph the original static graph.
 * \param topo_order topo order of the nodes in the graph.
 * \param node_importance The importance of the node
 * \param max_ncolor maximum number of colors allowed.
 * \param color the color index of each of the node.
 * \return the total number of colors.
 */
inline uint32_t ColorNodeGroup(
    const StaticGraph &graph,
    const std::vector<uint32_t> &topo_order,
    std::vector<uint32_t> node_importance,
    uint32_t max_ncolor,
    std::vector<uint32_t> *color) {
  CHECK_NE(max_ncolor, 0);
  CHECK_EQ(graph.nodes.size(), topo_order.size());
  CHECK_EQ(graph.nodes.size(), node_importance.size());

  color->clear();
  color->resize(topo_order.size(), max_ncolor);
  uint32_t cindex;
  // greedy algorithm, every time
  // find a path with best reward and assign a new color
  // All the nodes in the path cannot run in parallel.
  for (cindex = 0; cindex < max_ncolor - 1; ++cindex) {
    std::vector<uint32_t> path;
    uint32_t reward = FindBestPath(graph, topo_order, node_importance, &path);
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
  for (size_t i = 0; i < topo_order.size(); ++i) {
    if (color->at(i) == max_ncolor) {
      color->at(i) = cindex;
    }
  }
  return cindex + 1;
}

template <typename GNode, typename HashType, typename FVisit,
          typename HashFunc, typename InDegree, typename GetInput>
void PostOrderDFSVisit(const std::vector<GNode>& heads, FVisit fvisit,
                       HashFunc hash, InDegree indegree, GetInput getinput) {
  std::vector<std::pair<GNode, uint32_t> > stack;
  std::unordered_set<HashType> visited;
  for (auto& head : heads) {
    HashType head_hash = hash(head);
    if (visited.count(head_hash) == 0) {
      stack.push_back(std::make_pair(head, 0));
      visited.insert(head_hash);
    }
    while (!stack.empty()) {
      std::pair<GNode, uint32_t>& back = stack.back();
      if (back.second == indegree(back.first)) {
        fvisit(back.first);
        stack.pop_back();
      } else {
        const GNode& input = getinput(back.first, back.second++);
        HashType input_hash = hash(input);
        if (visited.count(input_hash) == 0) {
          stack.push_back(std::make_pair(input, 0));
          visited.insert(input_hash);
        }
      }
    }
  }
}

}  // namespace graph
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_ALGORITHM_H_
