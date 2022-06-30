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
 * \file build_subgraph.cc
 * \brief
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <unordered_set>
#include <stack>
#include <queue>

#include "./subgraph_property.h"
#include "mxnet/imperative.h"
#include "mxnet/base.h"

#define DEBUG_SUBGRAPH 0

namespace nnvm {
ObjectPtr CreateVariableNode(const std::string& name);
}

namespace mxnet {
namespace op {
namespace sg {  // sg stands for subgraph

#if DEBUG_SUBGRAPH
void PrintSubgraph(const std::vector<BiDirectedNode*>& simple_nodes) {
  std::string op_names = "";
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    op_names += simple_nodes[i]->node->attrs.name + ' ';
  }
  LOG(INFO) << "Subgraph node names: " << op_names;
}

void PrintNodeEntry(const nnvm::NodeEntry& entry) {
  std::string ret = "NodeEntry: node_name=" + entry.node->attrs.name +
                    ", index=" + std::to_string(entry.index) +
                    ", version=" + std::to_string(entry.version);
  LOG(INFO) << ret;
}

void PrintNodeEntries(const std::vector<nnvm::NodeEntry*>& entries) {
  for (size_t i = 0; i < entries.size(); ++i) {
    PrintNodeEntry(*entries[i]);
  }
}
#endif

/*!
 * \brief Given a MXNet computational graph, create an undirected graph from it.
 * \param g the MXNet computational graph
 * \param simple_nodes the nodes of undirected graph in top sorted order
 */
void CreateSimpleGraph(const nnvm::Graph& g, std::vector<BiDirectedNodePtr>* simple_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  simple_nodes->reserve(indexed_graph.num_nodes());
  DFSVisit(g.outputs, [&](const nnvm::ObjectPtr& node) {
    BiDirectedNodePtr sn = BiDirectedNode::Create();
    sn->node             = node.get();
    for (size_t i = 0; i < sn->node->inputs.size(); ++i) {
      const auto& e        = sn->node->inputs[i];
      const auto input_nid = indexed_graph.node_id(e.node.get());
      CHECK_LT(input_nid, simple_nodes->size());
      auto& input_node_outputs = (*simple_nodes)[input_nid]->outputs;
      auto it                  = input_node_outputs.find(sn->node);
      if (it == input_node_outputs.end()) {
        input_node_outputs.emplace(sn->node, std::vector<size_t>{i});
      } else {
        it->second.push_back(i);
      }
    }
    simple_nodes->emplace_back(std::move(sn));
  });
}

/*!
 * \brief Reset labels of the subgraph nodes to the original state
 * and clear the vector of subgraph nodes.
 */
void ResetNodeLabels(const nnvm::Graph& g,
                     const std::vector<BiDirectedNodePtr>& simple_nodes,
                     std::vector<BiDirectedNode*>* subgraph_nodes) {
  for (auto n : *subgraph_nodes) {
    const auto nid           = g.indexed_graph().node_id(n->node);
    simple_nodes[nid]->label = -1;
  }
  subgraph_nodes->clear();
}

/*!
 * \brief Prepare NodeAttr for node. NodeAttr will be used in SubgraphSelectorV2.
 */
static const std::shared_ptr<NodeAttr> PrepareNodeAttr(const nnvm::Graph& g,
                                                       const BiDirectedNode& node) {
  const auto& indexed_graph = g.indexed_graph();
  if (g.HasAttr("dtype") && g.HasAttr("shape") && g.HasAttr("dispatch_mode")) {
    const auto& vdtype         = g.GetAttr<nnvm::DTypeVector>("dtype");
    const auto& vshape         = g.GetAttr<mxnet::ShapeVector>("shape");
    const auto& dispatch_modes = g.GetAttr<mxnet::DispatchModeVector>("dispatch_mode");
    auto ret                   = std::make_shared<NodeAttr>();
    ret->dispatch_mode         = dispatch_modes[indexed_graph.node_id(node.node)];
    for (const auto& e : node.node->inputs) {
      ret->ishape.emplace_back(vshape[indexed_graph.entry_id(e)]);
      ret->itype.emplace_back(vdtype[indexed_graph.entry_id(e)]);
    }
    return ret;
  } else {
    return nullptr;
  }
}

/*!
 * \brief Given a subgraph, check if it has any external input entries.
 * \param g pointer to the whole graph.
 * \param simple_nods vector of simple nodes in top sorted order.
 * \param subgraph_nodes vector of pointers of simples of a subgraph.
 * \return true if the subgraph has external input, false otherwise.
 */
bool HasInputEntries(const nnvm::Graph& g,
                     const std::vector<BiDirectedNodePtr>& simple_nodes,
                     const std::vector<BiDirectedNode*>& subgraph_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  int label                 = -1;
  for (auto subgraph_node : subgraph_nodes) {
    if (label == -1) {
      label = subgraph_node->label;
    } else {
      CHECK_EQ(subgraph_node->label, label);
    }
    auto& inputs = subgraph_node->node->inputs;
    for (auto& e : inputs) {
      if (indexed_graph.exist(e.node.get())) {
        // e's source node is not a subgraph node
        const auto nid = indexed_graph.node_id(e.node.get());
        // this is a node not belonging to the subgraph
        if (simple_nodes[nid]->label != label) {
          return true;
        }
      } else {
        // e's source node is a subgraph node.
        // In this case, two subgraphs are adjacent.
        return true;
      }
    }
  }
  return false;
}

/*!
 * \brief This function traverses the nodes in a computation graph from a starting
 * node following the input edges and output edges, and marks all nodes that
 * can be accessed from the starting node. Before the function returns,
 * it will conduct checking whether there is a loop between the potential subgraph
 * and the outside nodes. If so, add the node that should break the loop
 * in excluded_nodes and return false. Otherwise, return true.
 * \param g the whole graph
 * \param subgraph_selector determines whether the visited node should be choosen or not
 * \param label the label of the current subgraph
 * \param snid node id of the seed simple node
 * \param simple_nodes all simple nodes in the top sorted order
 * \param subgraph_nodes all the nodes belonging to the same subgraph of seed node
 * \param excluded_nodes set of nodes that should be excluded from the current subgraph
 */
bool LabelSubgraph(const nnvm::Graph& g,
                   SubgraphSelectorV2Ptr subgraph_selector,
                   const int label,
                   const size_t snid,
                   const std::vector<BiDirectedNodePtr>& simple_nodes,
                   std::vector<BiDirectedNode*>* subgraph_nodes,
                   std::unordered_set<const BiDirectedNode*>* excluded_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  std::queue<BiDirectedNode*> node_queue;
  CHECK_EQ(simple_nodes[snid]->label, -1);
  simple_nodes[snid]->label = label;
  node_queue.push(simple_nodes[snid].get());
  // key: nodes that serve as input/output nodes to the subgraph
  // value: pair of vectors of nodes in the subgraph. The first vector contains the
  // output nodes of the key in the subgraph, and the second vector contains the
  // input nodes of the key in the subgraph.
  // If a non-subgraph node has inputs from the subgraph and the other non-subgraph node
  // has outputs to the subgraph, and the first non-subgraph node is an ancestor
  // of the second non-subgraph node, there exits a cycle.
  // When breaking the cycle, we want to start from removing the node with the largest node id
  // in the subgraph.
  std::unordered_map<const nnvm::Node*,
                     std::pair<std::vector<const nnvm::Node*>, std::vector<const nnvm::Node*>>>
      non_subgraph_node_map;
  while (!node_queue.empty()) {
    BiDirectedNode* cur_node = node_queue.front();
    node_queue.pop();
    subgraph_nodes->push_back(cur_node);
    // get qualified adjacent input nodes
    for (auto& e : cur_node->node->inputs) {
      const auto node = e.node.get();
      const auto nid  = indexed_graph.node_id(node);
      auto snode      = simple_nodes[nid].get();
      CHECK_LT(nid, simple_nodes.size());
      const bool select_input =
          (snode->label == -1) && (!excluded_nodes || !excluded_nodes->count(snode)) &&
          subgraph_selector->SelectInput(*cur_node, *snode, PrepareNodeAttr(g, *snode));
      if (select_input) {
        // e.node is a subgraph node
        snode->label = label;
        node_queue.push(snode);
      } else if (snode->label == -1) {
        // e.node is an input node of the subgraph
        non_subgraph_node_map[e.node.get()].first.push_back(cur_node->node);
      }
    }
    // get qualified output nodes
    for (auto it = cur_node->outputs.begin(); it != cur_node->outputs.end(); ++it) {
      const auto nid = indexed_graph.node_id(it->first);
      auto snode     = simple_nodes[nid].get();
      CHECK_LT(nid, simple_nodes.size());
      const bool select_output =
          (snode->label == -1) && (!excluded_nodes || !excluded_nodes->count(snode)) &&
          subgraph_selector->SelectOutput(*cur_node, *snode, PrepareNodeAttr(g, *snode));
      if (select_output) {
        // it->first is a subgraph node
        snode->label = label;
        node_queue.push(snode);
      } else if (snode->label == -1) {
        // it->first is an output node of the subgraph
        non_subgraph_node_map[it->first].second.push_back(cur_node->node);
      }
    }
  }
  // prepare to check if there is a cycle
  auto node_cmp = [&](const nnvm::Node* node1, const nnvm::Node* node2) {
    return indexed_graph.node_id(node1) < indexed_graph.node_id(node2);
  };
  std::vector<const nnvm::Node*> non_subgraph_nodes;
  non_subgraph_nodes.reserve(non_subgraph_node_map.size());
  for (auto& kv : non_subgraph_node_map) {
    auto& output_nodes = kv.second.first;
    std::sort(output_nodes.begin(), output_nodes.end(), node_cmp);
    auto& input_nodes = kv.second.second;
    std::sort(input_nodes.begin(), input_nodes.end(), node_cmp);
    non_subgraph_nodes.push_back(kv.first);
  }
  // check whether there is a cycle between the subgraph and its input/output nodes
  auto is_ancestor = [&](const nnvm::Node* ancestor,
                         const nnvm::Node* descendant,
                         const std::vector<BiDirectedNode*>& snodes) {
    if (ancestor == descendant)
      return true;
    std::unordered_set<nnvm::Node*> snode_set;
    for (const auto& sn : snodes) {
      snode_set.insert(sn->node);
    }
    std::stack<const nnvm::Node*> s;
    s.push(descendant);
    size_t count = 0;
    while (!s.empty() && count < indexed_graph.num_nodes()) {
      ++count;
      const nnvm::Node* top = s.top();
      s.pop();
      if (top == ancestor) {
        return true;
      }
      for (const auto& entry : top->inputs) {
        // when searching for the ancestor, the path cannot cross any subgraph node
        if (!snode_set.count(entry.node.get())) {
          s.push(entry.node.get());
        }
      }
    }
    return false;
  };
  std::sort(non_subgraph_nodes.begin(), non_subgraph_nodes.end(), node_cmp);
  int excluded_node_id = -1;
  for (size_t i = 0; i < non_subgraph_nodes.size(); ++i) {
    auto it1 = non_subgraph_node_map.find(non_subgraph_nodes[i]);
    CHECK(it1 != non_subgraph_node_map.end());
    auto& output_nodes = it1->second.first;   // has been top sorted
    auto& input_nodes  = it1->second.second;  // has been top sorted
    if (!output_nodes.empty() && !input_nodes.empty()) {
      // there is a loop between node i and the subgraph
      const auto node_id = std::max(indexed_graph.node_id(output_nodes.back()),
                                    indexed_graph.node_id(input_nodes.back()));
      excluded_node_id   = std::max(excluded_node_id, static_cast<int>(node_id));
    } else if (!input_nodes.empty()) {
      // node i is an input to the subgraph, find out if there is a node j
      // which is an output of the subgraph and also a child of node i.
      for (size_t j = i + 1; j < non_subgraph_nodes.size(); ++j) {
        auto it2 = non_subgraph_node_map.find(non_subgraph_nodes[j]);
        CHECK(it2 != non_subgraph_node_map.end());
        // i is topologically before j, j might be a direct/indirect output node of i
        CHECK_LT(indexed_graph.node_id(it1->first), indexed_graph.node_id(it2->first));
        if (!it2->second.first.empty() && is_ancestor(it1->first, it2->first, *subgraph_nodes)) {
          // found a loop
          const auto node_id = std::max(indexed_graph.node_id(input_nodes.back()),
                                        indexed_graph.node_id(it2->second.first.back()));
          excluded_node_id   = std::max(excluded_node_id, static_cast<int>(node_id));
        }
      }
    }
  }

  if (excluded_node_id != -1) {
    CHECK_LT(excluded_node_id, static_cast<int>(simple_nodes.size()));
    excluded_nodes->insert(simple_nodes[excluded_node_id].get());
    ResetNodeLabels(g, simple_nodes, subgraph_nodes);
    return false;
  }
  auto sim_node_cmp = [&](const BiDirectedNode* node1, const BiDirectedNode* node2) {
    return indexed_graph.node_id(node1->node) < indexed_graph.node_id(node2->node);
  };
  std::sort(subgraph_nodes->begin(), subgraph_nodes->end(), sim_node_cmp);
  return true;
}

/*!
 * \brief Finds all the nodes belonging to the same subgraph given a seed node.
 * \param g the whole graph
 * \subgraph_selector determines whether the visited node should be choosen or not
 * \label the label of the current subgraph
 * \snid node id of the seed simple node
 * \simple_nodes all simple nodes in the top sorted order
 * \subgraph_nodes all the nodes belonging to the same subgraph of seed node
 * \return Subgraph node candidates sorted in the topological order
 */
void PreSelectSubgraphNodes(const nnvm::Graph& g,
                            SubgraphSelectorV2Ptr subgraph_selector,
                            const int label,
                            const size_t snid,
                            const std::vector<BiDirectedNodePtr>& simple_nodes,
                            std::vector<BiDirectedNode*>* subgraph_nodes) {
  std::unordered_set<const BiDirectedNode*> excluded_nodes;
  size_t n_excluded_nodes    = 0;
  const size_t max_num_retry = simple_nodes.size() * simple_nodes.size();
  size_t count               = 0;
  bool success               = false;
  while (!success && count < max_num_retry) {
    success = LabelSubgraph(
        g, subgraph_selector, label, snid, simple_nodes, subgraph_nodes, &excluded_nodes);
    if (!success) {
      // Failed to label subgraph due to a cycle
      // If the number of excluded_nodes didn't change since the last iteration,
      // this means that there is no possible subgraph for the current node snid, we break
      // Otherwise, we keep trying (with the excluded nodes tagged)
      if (excluded_nodes.size() == n_excluded_nodes) {
        break;
      }
      n_excluded_nodes = excluded_nodes.size();
      std::string excluded_node_names;
      for (auto node : excluded_nodes) {
        excluded_node_names += node->node->attrs.name + ", ";
      }
      static int verbose = dmlc::GetEnv("MXNET_SUBGRAPH_VERBOSE", 1);
      if (verbose > 1) {
        LOG(INFO) << "Found a cycle when BFS from node " << simple_nodes[snid]->node->attrs.name
                  << ". Excluding nodes " << excluded_node_names << "and retrying";
      }
      subgraph_selector->Reset();
    }
    ++count;
  }
  if (success) {
    // check subgraph input. If none, reject the first op (in top order) from the subgraph
    // to make sure the subgraph gets external input.
    // this feature can be switched off by setting require_subgraph_inputs to false
    const SubgraphPropertyPtr& subg_prop = g.GetAttr<SubgraphPropertyPtr>("subgraph_property");
    if (subg_prop->HasAttr("require_subgraph_inputs") &&
        subg_prop->GetAttr<bool>("require_subgraph_inputs")) {
      if (subgraph_nodes->size() > 0 && !HasInputEntries(g, simple_nodes, *subgraph_nodes)) {
        // relabel the node to -1
        (*subgraph_nodes)[0]->label = -1;
        subgraph_nodes->erase(subgraph_nodes->begin());
      }
    }
  } else {
    LOG(INFO) << "Tried " << count << " times of finding subgraphs starting from node "
              << simple_nodes[snid]->node->attrs.name
              << " without success because a loop "
                 "is always found between the subgraph and some other nodes. Will treat "
                 "seed node "
              << simple_nodes[snid]->node->attrs.name << "as a subgraph with one node";
    CHECK(subgraph_nodes->empty());
    simple_nodes[snid]->label = label;
    subgraph_nodes->push_back(simple_nodes[snid].get());
  }
}

void SelectSubgraphNodes(nnvm::Graph* g,
                         SubgraphSelectorV2Ptr subgraph_selector,
                         const std::vector<BiDirectedNodePtr>& simple_nodes,
                         std::vector<std::vector<BiDirectedNode*>>* subgraph_nodes,
                         std::vector<SubgraphSelectorV2Ptr>* subgraph_selectors,
                         const BiDirectedNode* node,
                         const size_t snid,
                         size_t* subgraph_id) {
  const auto& indexed_graph = g->indexed_graph();

  auto node_cmp = [&](const BiDirectedNode* node1, const BiDirectedNode* node2) {
    return indexed_graph.node_id(node1->node) < indexed_graph.node_id(node2->node);
  };
  if ((simple_nodes[snid]->label == -1) &&
      subgraph_selector->Select(*node, PrepareNodeAttr(*g, *node))) {
    // pre-select nodes that can be grouped in a subgraph
    std::vector<BiDirectedNode*> preselected_nodes;
    PreSelectSubgraphNodes(
        *g, subgraph_selector, *subgraph_id, snid, simple_nodes, &preselected_nodes);

    // filter out unqualified pre-selected nodes
    std::vector<BiDirectedNode*> filtered_nodes = subgraph_selector->Filter(preselected_nodes);

    // reset node labels that are not in filtered nodes
    for (const auto n : preselected_nodes) {
      const auto nit = std::find(filtered_nodes.begin(), filtered_nodes.end(), n);
      if (nit == filtered_nodes.end()) {
        n->label = -1;
      }
    }

    if (filtered_nodes.size()) {
      // make sure filtered_nodes is a subset of preselected_nodes
      for (const auto n : filtered_nodes) {
        const auto nit = std::find(preselected_nodes.begin(), preselected_nodes.end(), n);
        CHECK(nit != preselected_nodes.end())
            << "Node " << n->node->attrs.name
            << " is not found in the pre-selected subgraph nodes."
               " Please make sure that no new nodes were added in your subgraph"
               " selector's Filter function";
      }

      // make sure nodes are sorted
      std::sort(filtered_nodes.begin(), filtered_nodes.end(), node_cmp);
      subgraph_nodes->push_back(filtered_nodes);
      subgraph_selectors->push_back(subgraph_selector);
      (*subgraph_id)++;
    }
  }
}

/*!
 * \brief Finds subgraphs with all nodes that meet certain criteria.
 * All nodes in a subgraph are marked with the same label.
 */
void FindSubgraphs(nnvm::Graph* g,
                   const SubgraphProperty& subg_prop,
                   const std::vector<BiDirectedNodePtr>& simple_nodes,
                   std::vector<std::vector<BiDirectedNode*>>* subgraph_nodes,
                   std::vector<SubgraphSelectorV2Ptr>* subgraph_selectors) {
  const auto& indexed_graph = g->indexed_graph();
  CHECK_EQ(indexed_graph.num_nodes(), simple_nodes.size());

  size_t subgraph_id = 0;
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    const auto snode                        = simple_nodes[i];
    SubgraphSelectorV2Ptr subgraph_selector = subg_prop.CreateSubgraphSelectorV2();
    SelectSubgraphNodes(g,
                        subgraph_selector,
                        simple_nodes,
                        subgraph_nodes,
                        subgraph_selectors,
                        snode.get(),
                        i,
                        &subgraph_id);
  }
}

/*!
 * \brief Sorts entries according to their topological order.
 * Note that entry ids cannot be used to sort entries.
 * \param entry_top_order_map mapping from entry pointer to its topological position in the graph
 * \param entries Node entries to be sorted
 */
void SortEntries(const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
                 std::vector<nnvm::NodeEntry*>* entries) {
  auto entry_cmp = [&](const nnvm::NodeEntry* e1, const nnvm::NodeEntry* e2) {
    const auto it1 = entry_top_order_map.find(e1);
    CHECK(it1 != entry_top_order_map.end());
    const auto it2 = entry_top_order_map.find(e2);
    CHECK(it2 != entry_top_order_map.end());
    return it1->second < it2->second;
  };
  std::sort(entries->begin(), entries->end(), entry_cmp);
}

/*!
 * \brief Given a subgraph, find the input entries of a subgraph.
 * \param g pointer to the whole graph
 * \param simple_nods vector of simple nodes in top sorted order
 * \param subgraph_nodes vector of pointers of simples of a subgraph.
 * \param entry_top_order_map mapping entry pointer to its top sorted position
 * \param input_entries input entries of the subgraph
 */
void FindInputEntries(const nnvm::Graph& g,
                      const std::vector<BiDirectedNodePtr>& simple_nodes,
                      const std::vector<BiDirectedNode*>& subgraph_nodes,
                      const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
                      std::vector<nnvm::NodeEntry*>* input_entries) {
  const auto& indexed_graph = g.indexed_graph();
  int label                 = -1;
  for (auto subgraph_node : subgraph_nodes) {
    if (label == -1) {
      label = subgraph_node->label;
    } else {
      CHECK_EQ(subgraph_node->label, label);
    }
    auto& inputs = subgraph_node->node->inputs;
    for (auto& e : inputs) {
      if (indexed_graph.exist(e.node.get())) {
        // e's source node is not a subgraph node
        const auto nid = indexed_graph.node_id(e.node.get());
        // this is a node not belonging to the subgraph
        if (simple_nodes[nid]->label != label) {
          input_entries->push_back(&e);
        }
      } else {
        // e's source node is a subgraph node.
        // In this case, two subgraphs are adjacent.
        input_entries->push_back(&e);
      }
    }
  }
  SortEntries(entry_top_order_map, input_entries);
}

/*!
 * \brief Given a subgraph, find the output entries of a subgraph.
 * \param g pointer to the whole graph
 * \param simple_nods vector of simple nodes in top sorted order
 * \param subgraph_nodes vector of pointers of simples of a subgraph.
 * \param entry_top_order_map mapping entry pointer to its top sorted position
 * \param output_entries output entries of the subgraph
 */
void FindOutputEntries(
    nnvm::Graph* g,
    const std::vector<BiDirectedNodePtr>& simple_nodes,
    const std::vector<BiDirectedNode*>& subgraph_nodes,
    const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
    std::vector<nnvm::NodeEntry*>* output_entries) {
  if (subgraph_nodes.empty())
    return;
  const auto& indexed_graph = g->indexed_graph();
  int label                 = -1;
  for (auto subgraph_node : subgraph_nodes) {
    if (label == -1) {
      label = subgraph_node->label;
    } else {
      CHECK_EQ(subgraph_node->label, label);
    }
    for (auto& output_node : subgraph_node->outputs) {
      if (indexed_graph.exist(output_node.first)) {
        // if the output node is a normal graph node (not a subgraph node)
        const auto nid = indexed_graph.node_id(output_node.first);
        // this is a node not belonging to the current subgraph
        if (simple_nodes[nid]->label != label) {
          for (auto idx : output_node.second) {
            auto& e = simple_nodes[nid]->node->inputs[idx];
            output_entries->push_back(&e);
          }
        }
      } else {
        // if the output node is a subgraph node
        // two graphs are adjacent
        for (auto idx : output_node.second) {
          output_entries->push_back(&(output_node.first->inputs[idx]));
        }
      }
    }
  }
  // Check if current subgraph contains a node which is the last node
  // of the whole graph. If so, save its corresponding entry as well.
  for (auto& entry : g->outputs) {
    // The entry might has been updated as an output of
    // a subgraph node. In this case, no need
    // to check its source for the current subgraph. Otherwise,
    // do the following.
    if (indexed_graph.exist(entry.node.get())) {
      const auto nid = indexed_graph.node_id(entry.node.get());
      if (simple_nodes[nid]->label == label) {
        output_entries->push_back(&entry);
      }
    }
  }
  SortEntries(entry_top_order_map, output_entries);
}

/*!
 * \brief Given a computation graph and a set of input node entries, this function cuts
 * the node entries and creates new variable nodes as the input nodes of the
 * subgraph. It returns the nodes that connect to the subgraph directly and
 * the names of the new variable nodes.
 */
void CutGraphInputs(const std::vector<nnvm::NodeEntry*>& input_entries,
                    std::vector<nnvm::NodeEntry>* orig_entries,
                    std::vector<nnvm::NodeEntry>* unique_orig_entries,
                    std::vector<nnvm::NodeEntry*>* unique_input_entries,
                    const bool skip_var = false,
                    const bool dedup    = false) {
  orig_entries->resize(input_entries.size());
  // map for creating unique var nodes for deduplicating entries from the same node
  std::unordered_map<std::string, nnvm::NodeEntry> name_map;
  std::unordered_map<std::string, int> name_count_map;

  for (size_t i = 0; i < input_entries.size(); ++i) {
    nnvm::NodeEntry* e = input_entries[i];
    // If the node is a variable itself, we may want to skip the node.
    if (e->node->is_variable() && skip_var) {
      continue;
    }
    // save all original entries
    orig_entries->at(i) = *e;
    // get unique name for this entry
    nnvm::Symbol sym;
    sym.outputs.push_back(*e);
    const auto output_names = sym.ListOutputNames();
    CHECK_EQ(output_names.size(), 1U);
    const std::string& var_name = output_names[0];
    // check if this entry is a duplicate
    if (name_count_map.count(var_name) == 0) {
      // first use of this node as input to subgraph
      name_count_map.emplace(var_name, 0);
      unique_orig_entries->push_back(*e);
      unique_input_entries->push_back(e);
      nnvm::ObjectPtr n = nnvm::CreateVariableNode(var_name + std::to_string(0));
      name_map.emplace(var_name, nnvm::NodeEntry{n, 0, 0});
    } else {
      // other use of same node as input to subgraph
      name_count_map[var_name]++;
    }

    if (dedup) {
      *e = name_map[var_name];
    } else {
      nnvm::ObjectPtr n =
          nnvm::CreateVariableNode(var_name + std::to_string(name_count_map[var_name]));
      *e = nnvm::NodeEntry{n, 0, 0};
    }
  }
}

/*!
 * \brief This function reattaches the original input nodes that were cut
 * by CutGraphInputs. This function is used when subgraphs are rejected, it
 * reattaches the subgraph back to the main graph where it was cut earlier.
 */
void ReattachGraphInputs(const std::vector<nnvm::NodeEntry*>& input_entries,
                         std::vector<nnvm::NodeEntry>* orig_entries) {
  for (size_t i = 0; i < input_entries.size(); ++i) {
    nnvm::NodeEntry* e = input_entries[i];
    *e                 = orig_entries->at(i);
  }
}

/*!
 * \brief Replace a set of nodes belonging to the same subgraph with a subgraph node
 * and keep the subgraph in the subgraph node.
 */
void CreateSubgraphNode(nnvm::Graph* g,
                        const std::vector<BiDirectedNodePtr>& simple_nodes,
                        const std::vector<BiDirectedNode*>& subgraph_nodes,
                        const SubgraphSelectorV2Ptr& subgraph_selector,
                        const size_t subgraph_id,
                        std::unordered_map<const nnvm::NodeEntry*, size_t>* entry_top_order_map) {
#if DEBUG_SUBGRAPH
  LOG(INFO) << "Searching for input entries...";
#endif
  bool dedup_subgraph = g->HasAttr("dedup_subgraph");
  std::vector<nnvm::NodeEntry*> input_entries;  // nodes that produce inputs to subgraph nodes
  FindInputEntries(*g, simple_nodes, subgraph_nodes, *entry_top_order_map, &input_entries);
  std::vector<nnvm::NodeEntry> orig_input_entries;     // original input entries (dupes)
  std::vector<nnvm::NodeEntry> unique_orig_entries;    // unique original input entries
  std::vector<nnvm::NodeEntry*> unique_input_entries;  // unique modified subgraph inputs
  CutGraphInputs(input_entries,
                 &orig_input_entries,
                 &unique_orig_entries,
                 &unique_input_entries,
                 false,
                 dedup_subgraph);
#if DEBUG_SUBGRAPH
  PrintNodeEntries(input_entries);
  LOG(INFO) << "Searching for output entries...";
#endif
  std::vector<nnvm::NodeEntry*> output_entries;
  FindOutputEntries(g, simple_nodes, subgraph_nodes, *entry_top_order_map, &output_entries);

  // Create a subgraph for the subgraph node
  // entries are in topological order, with duplicates being neighbors
  nnvm::Symbol sym;
  size_t idx = 0;
  nnvm::NodeEntryEqual node_equal;
  sym.outputs.resize(output_entries.size());
  for (size_t i = 0; i < output_entries.size(); ++i) {
    if (dedup_subgraph) {
      if (i == 0) {  // add first entry
        sym.outputs[idx] = *output_entries[i];
      } else if (!node_equal(sym.outputs[idx], *output_entries[i])) {  // compare to see if diff
        // add new entries
        idx++;
        sym.outputs[idx] = *output_entries[i];
      }  // else skip over dupe entries
    } else {
      sym.outputs[i] = *output_entries[i];
    }
  }
  if (dedup_subgraph)
    sym.outputs.resize(idx + 1);

  const SubgraphPropertyPtr& subg_prop = g->GetAttr<SubgraphPropertyPtr>("subgraph_property");
  if (dedup_subgraph)
    subg_prop->InitSubgraphInputs(&unique_input_entries, &unique_orig_entries);
  else
    subg_prop->InitSubgraphInputs(&input_entries, &orig_input_entries);
  nnvm::ObjectPtr n = subg_prop->CreateSubgraphNode(sym, subgraph_selector, subgraph_id);
  // CreateSubgraphNode returns NULL if subgraph property determines that subgraph is sub-optimal
  // In that case, subgraph node is not created and graph is not modified
  if (n) {
    // Connect the external nodes to the subgraph node.
    subg_prop->ConnectSubgraphOutputs(n, &output_entries);
    if (dedup_subgraph)
      subg_prop->ConnectSubgraphInputs(n, &unique_input_entries, &unique_orig_entries);
    else
      subg_prop->ConnectSubgraphInputs(n, &input_entries, &orig_input_entries);

    const auto& indexed_graph = g->indexed_graph();
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      auto& e = n->inputs[i];
      // update entry_top_order_map with newly created orig_input_entries
      auto it = entry_top_order_map->find(input_entries[i]);
      CHECK(it != entry_top_order_map->end());
      entry_top_order_map->emplace(&e, it->second);
      // update input entries' source simple nodes' outputs map
      nnvm::Node* node = e.node.get();
      if (indexed_graph.exist(node)) {
        const auto nid     = indexed_graph.node_id(node);
        BiDirectedNode* sn = simple_nodes[nid].get();
        for (BiDirectedNode* dest_node : subgraph_nodes) {
          sn->outputs.erase(dest_node->node);
        }
      }
    }

    // Set outputs according to current inputs
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      auto& e = n->inputs[i];
      // update input entries' source simple nodes' outputs map
      nnvm::Node* node = e.node.get();
      if (indexed_graph.exist(node)) {
        const auto nid     = indexed_graph.node_id(node);
        BiDirectedNode* sn = simple_nodes[nid].get();
        sn->outputs[n.get()].push_back(i);
      }
    }
  } else {
    ReattachGraphInputs(input_entries, &orig_input_entries);
  }
#if DEBUG_SUBGRAPH
  if (n)
    LOG(INFO) << "Subgraph node created and output_entries updated.";
  else
    LOG(INFO) << "Subgraph node not created, output_entries not updated.";
  PrintNodeEntries(output_entries);
#endif
}

/*!
 * \brief Adjust a set of nodes belonging to the same subgraph. No new node is created, but
 * adjust selected nodes' attributes.
 * This can be used to implement peephole optimization. For example, adjust calibration information
 * of quantized nodes.
 */
void AdjustSubgraphNode(nnvm::Graph* g,
                        const std::vector<BiDirectedNode*>& subgraph_nodes,
                        const SubgraphSelectorV2Ptr& subgraph_selector,
                        const size_t subgraph_id) {
  std::vector<nnvm::Node*> node_list;
  node_list.reserve(subgraph_nodes.size());
  for (auto node : subgraph_nodes) {
    node_list.push_back(node->node);
  }

  const SubgraphPropertyPtr& subg_prop = g->GetAttr<SubgraphPropertyPtr>("subgraph_property");
  subg_prop->AdjustSubgraphNode(node_list, subgraph_selector, subgraph_id);
}

}  // namespace sg

/*!
 * \brief Sort entries of all the nodes' inputs vectors in the topological order.
 * This is going to be used to sort input/output entries of subgraphs to keep
 * the topological order unchanged.
 */
void TopSortEntries(const nnvm::Graph& g,
                    std::unordered_map<const nnvm::NodeEntry*, size_t>* entry_top_order_map) {
  CHECK(entry_top_order_map != nullptr);
  std::unordered_set<const nnvm::Node*> visited;
  // tuple: (graph node, index of node's inputs, node entry as the output of the graph node)
  std::stack<std::tuple<nnvm::Node*, size_t, const nnvm::NodeEntry*>> s;
  auto in_degree = [](const nnvm::Node* node) -> size_t {
    if (!node) {
      return 0;
    }
    CHECK_EQ(node->control_deps.size(), 0U);
    return node->inputs.size();
  };
  for (auto& e : g.outputs) {
    nnvm::Node* node = e.node.get();
    if (visited.count(node) == 0U) {
      s.emplace(node, 0U, &e);
      visited.insert(node);
    } else {
      // The entry's source node has been visited before.
      // Marking the order for it.
      entry_top_order_map->emplace(&e, entry_top_order_map->size());
    }
    while (!s.empty()) {
      auto& top = s.top();
      if (std::get<1>(top) == in_degree(std::get<0>(top))) {
        // The node's inputs has been exhausted.
        entry_top_order_map->emplace(std::get<2>(top), entry_top_order_map->size());
        s.pop();
      } else {
        // The node still has input entries not visited.
        CHECK_LT(std::get<1>(top), std::get<0>(top)->inputs.size());
        auto& entry            = std::get<0>(top)->inputs[std::get<1>(top)++];
        nnvm::Node* input_node = entry.node.get();
        if (visited.count(input_node) == 0U) {
          // The entry's source node has not been visited.
          // Push the entry to the stack for marking order later.
          s.emplace(input_node, 0U, &entry);
          visited.insert(input_node);
        } else {
          // The entry's source node has been visited before.
          // Marking the order for it.
          entry_top_order_map->emplace(&entry, entry_top_order_map->size());
        }
      }
    }
  }
}

nnvm::Graph BuildSubgraph(nnvm::Graph&& g) {
  static int verbose = dmlc::GetEnv("MXNET_SUBGRAPH_VERBOSE", 1);
  if (!g.HasAttr("subgraph_property")) {  // treat the whole graph as a subgraph
    if (verbose > 1) {
      LOG(INFO) << "The graph has no attribute of subgraph_property attached. "
                   "The original graph is returned.";
    }
    return std::move(g);
  }
  using namespace sg;

  const SubgraphPropertyPtr& subg_prop = g.GetAttr<SubgraphPropertyPtr>("subgraph_property");
  if (verbose > 1) {
    const std::string& prop_name = subg_prop->HasAttr("property_name") ?
                                       subg_prop->GetAttr<std::string>("property_name") :
                                       "partition graph";
    LOG(INFO) << "start to execute " << prop_name << ".";
  }
  // top sort NodeEntry of all the nodes' inputs
  std::unordered_map<const nnvm::NodeEntry*, size_t> entry_top_order_map;
  TopSortEntries(g, &entry_top_order_map);

  // Create double directional graph for ease of finding subgraphs
  std::vector<BiDirectedNodePtr> simple_nodes;
  CreateSimpleGraph(g, &simple_nodes);
  std::vector<std::vector<BiDirectedNode*>> subgraph_nodes;
  std::vector<SubgraphSelectorV2Ptr> subgraph_selectors;
  FindSubgraphs(&g, *subg_prop, simple_nodes, &subgraph_nodes, &subgraph_selectors);
  CHECK_EQ(subgraph_nodes.size(), subgraph_selectors.size());
  for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
#if DEBUG_SUBGRAPH
    std::set<BiDirectedNode*> simple_node_set(subgraph_nodes[i].begin(), subgraph_nodes[i].end());
    CHECK_EQ(simple_node_set.size(), subgraph_nodes[i].size());
    PrintSubgraph(subgraph_nodes[i]);
#endif
    auto ptype = subg_prop->GetPropertyType();
    if (ptype == SubgraphProperty::SgPropertyType::kCreate) {
      CreateSubgraphNode(
          &g, simple_nodes, subgraph_nodes[i], subgraph_selectors[i], i, &entry_top_order_map);
    } else {
      CHECK_EQ(ptype, SubgraphProperty::SgPropertyType::kAdjust);
      AdjustSubgraphNode(&g, subgraph_nodes[i], subgraph_selectors[i], i);
    }
  }
  return std::move(g);
}

NNVM_REGISTER_PASS(BuildSubgraph)
    .describe(
        "Apply a subgraph pass according to the user defined rules "
        "in a derived class of SubgraphProperty")
    .set_body(BuildSubgraph)
    .set_change_graph(true);

}  // namespace op
}  // namespace mxnet
