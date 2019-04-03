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
 *  Copyright (c) 2018 by Contributors
 * \file partition_graph.cc
 * \brief
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>
#include <unordered_set>
#include <stack>
#include <queue>

#include "./subgraph_property.h"

namespace nnvm {
NodePtr CreateVariableNode(const std::string& name);
}

namespace mxnet {

namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::Graph;

#define DEBUG_SUBGRAPH 0

namespace sg {  // sg stands for subgraph

struct SimpleNode;
using SimpleNodePtr = std::shared_ptr<SimpleNode>;

/*!
 * \brief Node of the undirected graph which replicates the network structures
 * of the computational graph. It is used to ease the graph traversal for finding
 * subgraphs.
 */
struct SimpleNode {
  static SimpleNodePtr Create() {
    return std::make_shared<SimpleNode>();
  }
  SimpleNode() : label(-1), node(nullptr) {}
  /*! subgraph label */
  int label;
  /*! the original node in the computational graph it references*/
  nnvm::Node* node;
  /*!
   * \brief output nodes of the current node
   * key is node ptr and value is an array of indices standing for the entry indices
   * in key->inputs whose source is the current node.
   */
  std::unordered_map<nnvm::Node*, std::vector<size_t>> outputs;
};  // struct SimpleNode

#if DEBUG_SUBGRAPH
void PrintSubgraph(const std::vector<SimpleNode*>& simple_nodes) {
  std::string op_names = "";
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    op_names += simple_nodes[i]->node->attrs.name + ' ';
  }
  LOG(INFO) << "Subgraph node names: " << op_names;
}

void PrintNodeEntry(const nnvm::NodeEntry& entry) {
  std::string ret = "NodeEntry: node_name=" + entry.node->attrs.name
    + ", index=" + std::to_string(entry.index) + ", version=" + std::to_string(entry.version);
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
void CreateSimpleGraph(const Graph& g,
                       std::vector<SimpleNodePtr>* simple_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  simple_nodes->reserve(indexed_graph.num_nodes());
  DFSVisit(g.outputs, [&](const NodePtr& node) {
    SimpleNodePtr sn = SimpleNode::Create();
    sn->node = node.get();
    for (size_t i = 0; i < sn->node->inputs.size(); ++i) {
      const auto& e = sn->node->inputs[i];
      const auto input_nid = indexed_graph.node_id(e.node.get());
      CHECK_LT(input_nid, simple_nodes->size());
      auto& input_node_outputs = (*simple_nodes)[input_nid]->outputs;
      auto it = input_node_outputs.find(sn->node);
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
                     const std::vector<SimpleNodePtr>& simple_nodes,
                     std::vector<nnvm::Node*>* subgraph_nodes) {
  for (auto n : *subgraph_nodes) {
    const auto nid = g.indexed_graph().node_id(n);
    simple_nodes[nid]->label = -1;
  }
  subgraph_nodes->clear();
}

/*!
 * \brief This function traverses the nodes in a computation graph from a starting
 * node following the input edges and output edges, and marks all nodes that
 * can be accessed from the starting node. Before the function returns,
 * it will conduct checking whether there is a loop between the potential subgraph
 * and the outside nodes. If so, add the node that should break the loop
 * in excluded_nodes and return false. Otherwise, return true.
 * \param g the whole graph
 * \subgraph_selector determines whether the visited node should be choosen or not
 * \label the label of the current subgraph
 * \snid node id of the seed simple node
 * \simple_nodes all simple nodes in the top sorted order
 * \subgraph_nodes all the nodes belonging to the same subgraph of seed node
 * \excluded_nodes set of nodes that should be excluded from the current subgraph
 */
bool LabelSubgraph(const Graph& g,
                   SubgraphSelectorPtr subgraph_selector,
                   const int label,
                   const size_t snid,  // simple node id, this is a seed
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<nnvm::Node*>* subgraph_nodes,
                   std::unordered_set<const nnvm::Node*>* excluded_nodes = nullptr) {
  const auto& indexed_graph = g.indexed_graph();
  std::queue<SimpleNode*> node_queue;
  if (!excluded_nodes || !excluded_nodes->count(simple_nodes[snid]->node)) {
    CHECK_EQ(simple_nodes[snid]->label, -1);
    simple_nodes[snid]->label = label;
    node_queue.push(simple_nodes[snid].get());
  }
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
    std::pair<std::vector<const nnvm::Node*>,
              std::vector<const nnvm::Node*>>> non_subgraph_node_map;
  while (!node_queue.empty()) {
    SimpleNode* cur_node = node_queue.front();
    node_queue.pop();
    subgraph_nodes->push_back(cur_node->node);
    // get qualified adjacent input nodes
    for (auto& e : cur_node->node->inputs) {
      const bool select_input = (!excluded_nodes || !excluded_nodes->count(e.node.get()))
        && subgraph_selector->SelectInput(*cur_node->node, *e.node);
      if (select_input) {
        // e.node is a subgraph node
        const auto nid = indexed_graph.node_id(e.node.get());
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1) {
          simple_nodes[nid]->label = label;
          node_queue.push(simple_nodes[nid].get());
        }
      } else {
        // e.node is an input node of the subgraph
        non_subgraph_node_map[e.node.get()].first.push_back(cur_node->node);
      }
    }
    // get qualified output nodes
    for (auto it = cur_node->outputs.begin(); it != cur_node->outputs.end(); ++it) {
      const bool select_output = (!excluded_nodes || !excluded_nodes->count(it->first))
          && subgraph_selector->SelectOutput(*cur_node->node, *it->first);
      if (select_output) {
        // it->first is a subgraph node
        const auto nid = indexed_graph.node_id(it->first);
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1) {
          simple_nodes[nid]->label = label;
          node_queue.push(simple_nodes[nid].get());
        }
      } else {
        // it->first is an output node of the subgraph
        non_subgraph_node_map[it->first].second.push_back(cur_node->node);
      }
    }
  }
  // prepare to check if there is a cycle
  auto node_cmp = [&] (const nnvm::Node* node1, const nnvm::Node* node2) {
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
  auto is_ancestor = [&](const nnvm::Node* ancestor, const nnvm::Node* descendant,
                         const std::vector<nnvm::Node*>& snodes) {
    if (ancestor == descendant) return true;
    std::stack<const nnvm::Node*> s;
    s.push(descendant);
    size_t count = 0;
    while (!s.empty()) {
      CHECK_LT(count, indexed_graph.num_nodes()) << "Finding ancestor failed. There is probably"
                                                    " a loop in the graph";
      ++count;
      const nnvm::Node* top = s.top();
      s.pop();
      if (top == ancestor) {
        return true;
      }
      for (const auto& entry : top->inputs) {
        // when searching for the ancestor, the path cannot cross any subgraph node
        auto it = std::find(snodes.begin(), snodes.end(), entry.node.get());
        if (it == snodes.end()) {
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
    auto& output_nodes = it1->second.first;  // has been top sorted
    auto& input_nodes = it1->second.second;  // has been top sorted
    if (!output_nodes.empty() && !input_nodes.empty()) {
      // there is a loop between node i and the subgraph
      const auto node_id = std::max(indexed_graph.node_id(output_nodes.back()),
                                    indexed_graph.node_id(input_nodes.back()));
      excluded_node_id = std::max(excluded_node_id, static_cast<int>(node_id));
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
          excluded_node_id = std::max(excluded_node_id, static_cast<int>(node_id));
        }
      }
    }
  }

  if (excluded_node_id != -1) {
    CHECK_LT(excluded_node_id, static_cast<int>(simple_nodes.size()));
    CHECK_NE(excluded_node_id, static_cast<int>(snid))
      << "A cycle is found in the computational graph between nodes "
      << simple_nodes[excluded_node_id]->node->attrs.name << " and "
      << simple_nodes[snid]->node->attrs.name;
    excluded_nodes->insert(simple_nodes[excluded_node_id]->node);
    ResetNodeLabels(g, simple_nodes, subgraph_nodes);
    return false;
  }
  std::sort(subgraph_nodes->begin(), subgraph_nodes->end(), node_cmp);
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
void PreSelectSubgraphNodes(const Graph& g,
                            SubgraphSelectorPtr subgraph_selector,
                            const int label,
                            const size_t snid,
                            const std::vector<SimpleNodePtr>& simple_nodes,
                            std::vector<nnvm::Node*>* subgraph_nodes) {
  std::unordered_set<const nnvm::Node*> excluded_nodes;
  const size_t max_num_retry = simple_nodes.size() * simple_nodes.size();
  size_t count = 0;
  bool success = false;
  while (!success && count < max_num_retry) {
    success = LabelSubgraph(g, subgraph_selector, label, snid, simple_nodes,
                            subgraph_nodes, &excluded_nodes);
    if (!success) {
      CHECK(!excluded_nodes.empty());
      std::string excluded_node_names;
      for (auto node : excluded_nodes) {
        excluded_node_names += node->attrs.name + ", ";
      }
      LOG(INFO) << "Found a cycle when BFS from node " << simple_nodes[snid]->node->attrs.name
                << ". Excluding nodes " << excluded_node_names << "and retrying";
    }
    ++count;
  }
  if (!success) {
    LOG(INFO) << "Tried " << count << " times of finding subgraphs starting from node "
              << simple_nodes[snid]->node->attrs.name << " without success because a loop "
                  "is always found between the subgraph and some other nodes. Will treat "
                  "seed node " << simple_nodes[snid]->node->attrs.name
              << "as a subgraph with one node";
    CHECK(subgraph_nodes->empty());
    simple_nodes[snid]->label = label;
    subgraph_nodes->push_back(simple_nodes[snid]->node);
  }
}

/*!
 * \brief Given a vector of nodes, group them into individual subgraphs
 * based upon their connectivity.
 */
void PostProcessNodeCandidates(const nnvm::Graph& g,
                               const std::vector<nnvm::Node*>& nodes,
                               const std::vector<SimpleNodePtr>& simple_nodes,
                               std::vector<std::vector<SimpleNode*>>* subgraphs,
                               size_t* subgraph_id) {
  const auto& indexed_graph = g.indexed_graph();
  std::unordered_set<nnvm::Node*> node_set(nodes.begin(), nodes.end());
  auto simple_node_cmp = [&] (const SimpleNode* node1, const SimpleNode* node2) {
    return indexed_graph.node_id(node1->node) < indexed_graph.node_id(node2->node);
  };
  for (auto node : nodes) {
    if (!node_set.count(node)) {
      // The node has been included in a subgraph
      continue;
    }
    std::queue<nnvm::Node*> q;
    q.push(node);
    CHECK_EQ(node_set.erase(node), 1U);
    subgraphs->emplace_back();
    const auto nid = indexed_graph.node_id(node);
    simple_nodes[nid]->label = *subgraph_id;
    subgraphs->back().push_back(simple_nodes[nid].get());
    while (!q.empty()) {
      nnvm::Node* cur_node = q.front();
      q.pop();
      for (auto& e : cur_node->inputs) {
        auto in_it = node_set.find(e.node.get());
        if (in_it != node_set.end()) {
          q.push(*in_it);
          const auto in_nid = indexed_graph.node_id(*in_it);
          simple_nodes[in_nid]->label = *subgraph_id;
          subgraphs->back().push_back(simple_nodes[in_nid].get());
          node_set.erase(in_it);
        }
      }
      const auto cur_nid = indexed_graph.node_id(cur_node);
      const SimpleNode* cur_snode = simple_nodes[cur_nid].get();
      for (const auto& kv : cur_snode->outputs) {
        const auto out_it = node_set.find(kv.first);
        if (out_it != node_set.end()) {
          q.push(*out_it);
          const auto out_nid = indexed_graph.node_id(*out_it);
          simple_nodes[out_nid]->label = *subgraph_id;
          subgraphs->back().push_back(simple_nodes[out_nid].get());
          node_set.erase(out_it);
        }
      }
    }
    ++(*subgraph_id);
    std::sort(subgraphs->back().begin(), subgraphs->back().end(), simple_node_cmp);
  }
  CHECK(node_set.empty());
}

/*!
 * \brief Finds subgraphs with all nodes that meet certain criteria.
 * All nodes in a subgraph are marked with the same label.
 */
void FindSubgraphs(Graph* g,
                   const SubgraphProperty &subg_prop,
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<std::vector<SimpleNode*>>* subgraph_nodes) {
  const auto& indexed_graph = g->indexed_graph();
  CHECK_EQ(indexed_graph.num_nodes(), simple_nodes.size());
  auto node_cmp = [&] (const nnvm::Node* node1, const nnvm::Node* node2) {
    return indexed_graph.node_id(node1) < indexed_graph.node_id(node2);
  };
  size_t subgraph_id = 0;
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    nnvm::Node* node = simple_nodes[i]->node;
    auto subgraph_selector = subg_prop.CreateSubgraphSelector();
    if (subgraph_selector->Select(*node) && simple_nodes[i]->label == -1) {
      // pre-select nodes that can be grouped in a subgraph
      std::vector<nnvm::Node*> preselected_nodes;
      PreSelectSubgraphNodes(*g, subgraph_selector, subgraph_id, i, simple_nodes,
                             &preselected_nodes);

      // filter out unqualified pre-selected nodes
      std::vector<nnvm::Node*> filtered_nodes = subgraph_selector->Filter(preselected_nodes);

      // make sure filtered_nodes is a subset of preselected_nodes
      for (const auto n : filtered_nodes) {
        const auto nit = std::find(preselected_nodes.begin(), preselected_nodes.end(), n);
        CHECK(nit != preselected_nodes.end())
          << "Node " << n->attrs.name << " is not found in the pre-selected subgraph nodes."
             " Please make sure that no new nodes were added in your subgraph"
             " selector's Filter function";
      }

      // make sure nodes are sorted
      std::sort(filtered_nodes.begin(), filtered_nodes.end(), node_cmp);

      // reset node labels that are not in filtered nodes
      for (const auto n : preselected_nodes) {
        const auto nit = std::find(filtered_nodes.begin(), filtered_nodes.end(), n);
        if (nit == filtered_nodes.end()) {
          simple_nodes[indexed_graph.node_id(n)]->label = -1;
        }
      }
      // find out subgraphs from the filtered nodes
      std::vector<std::vector<SimpleNode*>> subgraphs;
      PostProcessNodeCandidates(*g, filtered_nodes, simple_nodes, &subgraphs, &subgraph_id);
      if (!subgraphs.empty()) {
        subgraph_nodes->insert(subgraph_nodes->end(), subgraphs.begin(), subgraphs.end());
      }
    }
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
 * \brief Given a subgraph, find the output entries of a subgraph.
 * \param g pointer to the whole graph
 * \param simple_nods vector of simple nodes in top sorted order
 * \param subgraph_nodes vector of pointers of simples of a subgraph.
 * \param entry_top_order_map mapping entry pointer to its top sorted position
 * \param input_entries input entries of the subgraph
 */
void FindInputEntries(const Graph& g,
                      const std::vector<SimpleNodePtr>& simple_nodes,
                      const std::vector<SimpleNode*>& subgraph_nodes,
                      const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
                      std::vector<nnvm::NodeEntry*>* input_entries) {
  const auto& indexed_graph = g.indexed_graph();
  int label = -1;
  for (auto subgraph_node : subgraph_nodes) {
    if (label == -1) {
      label = subgraph_node->label;
    } else {
      CHECK_EQ(subgraph_node->label, label);
    }
    auto& inputs = subgraph_node->node->inputs;
    for (auto &e : inputs) {
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
void FindOutputEntries(Graph* g,
                       const std::vector<SimpleNodePtr>& simple_nodes,
                       const std::vector<SimpleNode*>& subgraph_nodes,
                       const std::unordered_map<const nnvm::NodeEntry*, size_t>&
                         entry_top_order_map,
                       std::vector<nnvm::NodeEntry*>* output_entries) {
  if (subgraph_nodes.empty()) return;
  const auto& indexed_graph = g->indexed_graph();
  int label = -1;
  for (auto subgraph_node : subgraph_nodes) {
    if (label == -1) {
      label = subgraph_node->label;
    } else {
      CHECK_EQ(subgraph_node->label, label);
    }
    for (auto &output_node : subgraph_node->outputs) {
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
  for (auto &entry : g->outputs) {
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
void CutGraphInputs(const std::vector<nnvm::NodeEntry*> &input_entries,
                    std::vector<nnvm::NodeEntry> *orig_entries,
                    const bool skip_var = false) {
  orig_entries->resize(input_entries.size());
  // map for creating unique var nodes for deduplicating entries from the same node
  std::unordered_map<std::string, int> name_count_map;
  for (size_t i = 0; i < input_entries.size(); ++i) {
    nnvm::NodeEntry *e = input_entries[i];
    // If the node is a variable itself, we may want to skip the node.
    if (e->node->is_variable() && skip_var) {
      continue;
    }

    orig_entries->at(i) = *e;
    nnvm::Symbol sym;
    sym.outputs.push_back(*e);
    const auto output_names = sym.ListOutputNames();
    CHECK_EQ(output_names.size(), 1U);
    const std::string& var_name = output_names[0];
    auto it = name_count_map.find(var_name);
    if (name_count_map.end() == it) {
      name_count_map.emplace(var_name, 0);
    } else {
      ++(it->second);
    }
    nnvm::NodePtr n = nnvm::CreateVariableNode(var_name + std::to_string(name_count_map[var_name]));
    *e = nnvm::NodeEntry{n, 0, 0};
  }
}

/*!
 * \brief Replace a set of nodes belonging to the same subgraph with a subgrpah node
 * and keep the subgraph in the subgraph node. The input entries and output entries
 * of the subgraph node are kept in the same order as the subgraph's.
 */
void CreateSubgraphNode(Graph* g,
                        const std::vector<SimpleNodePtr>& simple_nodes,
                        const std::vector<SimpleNode*>& subgraph_nodes,
                        const size_t subgraph_id,
                        std::unordered_map<const nnvm::NodeEntry*, size_t>* entry_top_order_map) {
#if DEBUG_SUBGRAPH
  LOG(INFO) << "Searching for input entries...";
#endif
  std::vector<nnvm::NodeEntry*> input_entries;
  FindInputEntries(*g, simple_nodes, subgraph_nodes, *entry_top_order_map, &input_entries);
  std::vector<nnvm::NodeEntry> orig_input_entries;
  CutGraphInputs(input_entries, &orig_input_entries, false);
#if DEBUG_SUBGRAPH
  PrintNodeEntries(input_entries);
  LOG(INFO) << "Searching for output entries...";
#endif
  std::vector<nnvm::NodeEntry*> output_entries;
  FindOutputEntries(g, simple_nodes, subgraph_nodes, *entry_top_order_map, &output_entries);

  // Create a subgraph for the subgraph node
  nnvm::Symbol sym;
  sym.outputs.resize(output_entries.size());
  for (size_t i = 0; i < output_entries.size(); ++i) {
    sym.outputs[i] = *output_entries[i];
  }
  const SubgraphPropertyPtr& subg_prop = g->GetAttr<SubgraphPropertyPtr>("subgraph_property");
  nnvm::NodePtr n = subg_prop->CreateSubgraphNode(sym, subgraph_id);

  // Connect the external nodes to the subgraph node.
  subg_prop->ConnectSubgraphOutputs(n, &output_entries);
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
      const auto nid = indexed_graph.node_id(node);
      SimpleNode* sn = simple_nodes[nid].get();
      for (SimpleNode* dest_node : subgraph_nodes) {
        sn->outputs.erase(dest_node->node);
      }
      sn->outputs[n.get()].push_back(i);
    }
  }
#if DEBUG_SUBGRAPH
  PrintNodeEntries(output_entries);
#endif
}

}  // namespace sg

/*!
 * \brief Sort entries of all the nodes' inputs vectors in the topological order.
 * This is going to be used to sort input/output entries of subgraphs to keep
 * the topological order unchanged.
 */
void TopSortEntries(const Graph& g,
                    std::unordered_map<const nnvm::NodeEntry*, size_t>* entry_top_order_map) {
  CHECK(entry_top_order_map != nullptr);
  std::unordered_set<const nnvm::Node*> visited;
  // tuple: (graph node, index of node's inputs, node entry as the output of the graph node)
  std::stack<std::tuple<nnvm::Node*, size_t, const nnvm::NodeEntry*>> s;
  auto in_degree = [] (const nnvm::Node* node)->size_t {
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
        auto& entry = std::get<0>(top)->inputs[std::get<1>(top)++];
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

Graph PartitionGraph(Graph&& g) {
  if (!g.HasAttr("subgraph_property")) {  // treat the whole graph as a subgraph
    LOG(INFO) << "The graph has no attribute of subgraph_property attached. "
                 "The original graph is returned.";
    return g;
  }
  using namespace sg;
  const SubgraphPropertyPtr& subg_prop = g.GetAttr<SubgraphPropertyPtr>("subgraph_property");
  // top sort NodeEntry of all the nodes' inputs
  std::unordered_map<const nnvm::NodeEntry*, size_t> entry_top_order_map;
  TopSortEntries(g, &entry_top_order_map);

  // Create undirected graph for ease of finding subgraphs
  std::vector<SimpleNodePtr> simple_nodes;
  CreateSimpleGraph(g, &simple_nodes);
  std::vector<std::vector<SimpleNode*>> subgraph_nodes;
  FindSubgraphs(&g, *subg_prop, simple_nodes, &subgraph_nodes);
  for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
#if DEBUG_SUBGRAPH
    std::set<SimpleNode*> simple_node_set(subgraph_nodes[i].begin(), subgraph_nodes[i].end());
    CHECK_EQ(simple_node_set.size(), subgraph_nodes[i].size());
    PrintSubgraph(subgraph_nodes[i]);
#endif
    CreateSubgraphNode(&g, simple_nodes, subgraph_nodes[i], i, &entry_top_order_map);
  }
  return g;
}

NNVM_REGISTER_PASS(PartitionGraph)
.describe("Partition a graph according to the user defined rules "
          "in a derived class of SubgraphProperty")
.set_body(PartitionGraph)
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
