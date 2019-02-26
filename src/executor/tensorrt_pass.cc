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
 * Copyright (c) 2018 by Contributors
 * \file tensorrt_pass.cc
 * \brief Replace TRT compatible subgraphs by TRT engines
 * \author Clement Fuji Tsang
 */

#if MXNET_USE_TENSORRT

#include <NvInfer.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <nnvm/graph_attr_types.h>
#include <onnx/onnx_pb.h>

#include "../operator/contrib/nnvm_to_onnx-inl.h"
#include "./exec_pass.h"
#include "./onnx_to_tensorrt.h"

namespace mxnet {
namespace exec {

using NodePtr = nnvm::NodePtr;

/*!
 * \brief Custom graph class, which will contain bi-directional nodes
 * we need to compute DFS and reverse DFS for graph partitioning
 */
class BidirectionalGraph {
 public:
  struct Node {
    nnvm::Node* nnvmptr;
    std::vector<Node*> inputs;
    std::vector<Node*> outputs;
  };
  std::vector<Node> nodes;
  std::unordered_map<nnvm::Node*, uint32_t> nnvm2nid;
  std::vector<Node*> outputs;
  static const std::unordered_set<std::string> unconditionalTRTop;

  explicit BidirectionalGraph(const Graph &g) {
    auto& idx = g.indexed_graph();
    auto num_nodes = idx.num_nodes();
    nodes.reserve(num_nodes);
    nnvm2nid.reserve(num_nodes);
    outputs.reserve(idx.outputs().size());
    DFSVisit(g.outputs, [this](const nnvm::NodePtr& n) {
      BidirectionalGraph::Node new_node;
      new_node.nnvmptr = n.get();
      nnvm2nid[n.get()] = static_cast<uint32_t>(nodes.size());
      nodes.emplace_back(std::move(new_node));
    });
    for (const auto& it : nnvm2nid) {
      nnvm::Node* nnvmnode = it.first;
      uint32_t nid = it.second;
      for (auto& n : nnvmnode->inputs) {
        uint32_t input_nid = nnvm2nid[n.node.get()];
        nodes[input_nid].outputs.emplace_back(&nodes[nid]);
        nodes[nid].inputs.emplace_back(&nodes[input_nid]);
      }
    }
    for (auto& e : g.outputs) {
      uint32_t nid = nnvm2nid[e.node.get()];
      outputs.emplace_back(&nodes[nid]);
    }
  }

  template <typename FVisit>
  void DFS(const std::vector<Node*>& heads, bool reverse, FVisit fvisit) {
    std::unordered_set<Node*> visited;
    std::vector<Node*> vec(heads.begin(), heads.end());
    visited.reserve(heads.size());
    while (!vec.empty()) {
      Node* vertex = vec.back();
      vec.pop_back();
      if (visited.count(vertex) == 0) {
        visited.insert(vertex);
        fvisit(vertex);
        std::vector<Node*> nexts = reverse ? vertex->inputs : vertex->outputs;
        for (Node* node : nexts) {
          if (visited.count(node) == 0) {
            vec.emplace_back(node);
          }
        }
      }
    }
  }

  using t_pairset = std::pair<std::unordered_set<Node*>, std::unordered_set<Node*>>;
  using t_pairvec = std::pair<std::vector<Node*>, std::vector<Node*>>;
  using t_uncomp_map = std::unordered_map<Node*, std::unordered_set<Node*>>;

  std::unordered_set<Node*> naive_grow_subgraph(Node* head,
                                                std::unordered_set<Node*>* set_unused,
                                                t_uncomp_map* uncomp_map) {
    std::unordered_set<Node*> subgraph;
    std::unordered_set<Node*> uncomp_set;
    std::deque<Node*> stack;
    stack.emplace_back(head);
    while (!stack.empty()) {
      Node* vertex = stack.back();
      stack.pop_back();
      if (set_unused->count(vertex) && !uncomp_set.count(vertex)) {
        set_unused->erase(vertex);
        subgraph.insert(vertex);
        uncomp_set.insert((*uncomp_map)[vertex].begin(), (*uncomp_map)[vertex].end());
        for (Node* input : vertex->inputs) {
          if (set_unused->count(input) && !uncomp_set.count(input)) {
            stack.emplace_back(input);
          }
        }
        for (Node* output : vertex->outputs) {
          if (set_unused->count(output) && !uncomp_set.count(output)) {
            stack.emplace_back(output);
          }
        }
      }
    }
    return subgraph;
  }

  std::vector<std::unordered_set<Node*>> get_subsets(
    std::unordered_map<std::string, NDArray>* const params_map) {
    std::vector<std::unordered_set<Node*>> subgraphs;
    std::unordered_set<Node*> set_nonTRTnodes;
    std::unordered_set<Node*> set_allnodes(nodes.size());
    std::vector<t_pairset> separation_sets;
    for (Node& node : nodes) {
      if (!IsTRTCompatible(node.nnvmptr)) {
        set_nonTRTnodes.insert(&node);
        std::unordered_set<Node*> in_graph;
        std::unordered_set<Node*> out_graph;
        std::vector<Node*> dummy_head;
        dummy_head.emplace_back(&node);
        DFS(dummy_head, false, [&out_graph](Node* node) {
          out_graph.insert(node);
        });
        DFS(dummy_head, true, [&in_graph](Node* node) {
          in_graph.insert(node);
        });
        separation_sets.emplace_back(std::make_pair(in_graph, out_graph));
      }
      set_allnodes.emplace(&node);
    }
    t_uncomp_map uncomp_map;
    std::unordered_set<Node*> set_TRTnodes;
    set_TRTnodes.insert(set_allnodes.begin(), set_allnodes.end());
    for (Node* n : set_nonTRTnodes) {
      set_TRTnodes.erase(n);
    }
    for (Node* n : set_TRTnodes) {
      for (t_pairset p : separation_sets) {
        if (p.first.count(n)) {
          uncomp_map[n].insert(p.second.begin(), p.second.end());
        } else if (p.second.count(n)) {
          uncomp_map[n].insert(p.first.begin(), p.first.end());
        }
      }
      for (Node* nonTRTn : set_nonTRTnodes) {
        uncomp_map[n].erase(nonTRTn);
      }
    }
    std::unordered_set<Node*> set_unused;
    set_unused.reserve(set_TRTnodes.size());

    for (auto& n : set_TRTnodes) {
      if (n->nnvmptr->attrs.op != nullptr || params_map->count(n->nnvmptr->attrs.name)) {
        set_unused.insert(n);
      }
    }
    std::unordered_set<Node*> visited;
    std::deque<Node*> stack(outputs.begin(), outputs.end());
    while (!stack.empty()) {
      Node* vertex = stack.front();
      stack.pop_front();
      if (!visited.count(vertex)) {
        visited.insert(vertex);
        if (set_unused.count(vertex)) {
          subgraphs.emplace_back(naive_grow_subgraph(vertex, &set_unused, &uncomp_map));
        }
        for (Node* input : vertex->inputs) {
          stack.emplace_back(input);
        }
      }
    }

    return subgraphs;
  }


 private:
  friend class Graph;

  bool IsTRTCompatible(nnvm::Node* nodeptr) {
    if (nodeptr->op() == nullptr) {
      return true;
    }

    const std::string op_name = nodeptr->op()->name;
    if (op_name == "Pooling") {
      return (nodeptr->attrs.dict.at("pool_type") == "avg" ||
          nodeptr->attrs.dict.at("pool_type") == "max");
    }

    if (unconditionalTRTop.count(op_name)) {
      return true;
    }

    if (op_name == "Activation") {
      return nodeptr->attrs.dict.at("act_type") == "relu" ||
        nodeptr->attrs.dict.at("act_type") == "tanh" ||
        nodeptr->attrs.dict.at("act_type") == "sigmoid";
    }

    return false;
  }
};  // class BidirectionalGraph

/*!
 * \brief function which transform std::vector<dmlc::any> back to Attrs (dmlc::any)
 */
const std::unordered_set<std::string> BidirectionalGraph::unconditionalTRTop = {
  "Convolution",
  "BatchNorm",
  "elemwise_add",
  "elemwise_sub",
  "elemwise_mul",
  "rsqrt",
  "pad",
  "Pad",
  "mean",
  "FullyConnected",
  "Flatten",
  "SoftmaxOutput",
};


using NodeEntrySet = std::unordered_set<nnvm::NodeEntry, nnvm::NodeEntryHash,
                                        nnvm::NodeEntryEqual>;

/*!
 * \brief get the output nodes of the subgraph in the main graph
 * \return a vector of the output nodes
*/
std::vector<nnvm::NodeEntry> GetSubgraphNodeEntries(Graph g,
    std::unordered_set<nnvm::Node*> set_subgraph) {
  std::vector<nnvm::NodeEntry> outputs;
  NodeEntrySet _outputs;
  for (auto& e : g.outputs) {
    if (set_subgraph.count(e.node.get())) {
      _outputs.insert(e);
    }
  }
  DFSVisit(g.outputs, [&set_subgraph, &_outputs](const nnvm::NodePtr &node){
    if (!set_subgraph.count(node.get())) {
      for (auto& e : node->inputs) {
        if (set_subgraph.count(e.node.get())) {
          _outputs.insert(e);
        }
      }
    }
  });
  outputs.insert(outputs.begin(), _outputs.begin(), _outputs.end());
  return outputs;
}


/*!
 * \brief get the nodes outside of the subgraph for which outputs are used in the subgraph
 * \return a vector the nodes
*/
std::vector<nnvm::NodeEntry> GetSubgraphInterfaceNodes(Graph g,
    std::unordered_set<nnvm::Node*> set_subgraph) {
  std::vector<nnvm::NodeEntry> inputs;
  NodeEntrySet _inputs;
  DFSVisit(g.outputs, [&set_subgraph, &_inputs](const nnvm::NodePtr &node){
    if (set_subgraph.count(node.get())) {
      for (auto& e : node->inputs) {
        if (!set_subgraph.count(e.node.get())) {
          _inputs.insert(e);
        }
      }
    }
  });
  inputs.insert(inputs.begin(), _inputs.begin(), _inputs.end());
  return inputs;
}

std::unordered_map<uint32_t, uint32_t> GetGraphInputsMap(const Graph& g) {
  std::unordered_map<uint32_t, uint32_t> outputs;
  auto& idx = g.indexed_graph();
  outputs.reserve(idx.num_nodes());
  std::vector<uint32_t> input_nodes = idx.input_nodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    outputs[input_nodes[i]] = static_cast<uint32_t>(i);
  }
  return outputs;
}

/*!
 * \brief Dummy function which creates a fake TensorRT Node
 */
nnvm::NodePtr ConvertNnvmGraphToOnnx(const nnvm::Graph &g,
                                     std::unordered_map<std::string, NDArray>* const params_map) {
  auto p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_trt_op");
  op::ONNXParam onnx_param = op::nnvm_to_onnx::ConvertNnvmGraphToOnnx(g, params_map);
  p->attrs.dict["serialized_output_map"] = onnx_param.serialized_output_map;
  p->attrs.dict["serialized_input_map"]  = onnx_param.serialized_input_map;
  p->attrs.dict["serialized_onnx_graph"] = onnx_param.serialized_onnx_graph;
  if (p->op()->attr_parser != nullptr) {
    p->op()->attr_parser(&(p->attrs));
  }
  return p;
}

/*!
 * \brief Update attributes of the graph (such as some inputs properties)
 */
Graph UpdateSubgraphAttrs(Graph&& subgraph, const Graph& g,
                          const std::unordered_map<nnvm::Node*, nnvm::NodePtr>& old2new,
                          const nnvm::NodeEntryMap<nnvm::NodeEntry>& main_input_entry_to_sub) {
  const auto& idx     = g.indexed_graph();
  const auto& sub_idx = subgraph.indexed_graph();

  const auto& shape               = g.GetAttr<nnvm::ShapeVector>("shape");
  const auto& dtype               = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& storage_type        = g.GetAttr<StorageTypeVector>("storage_type");
  const auto& shape_inputs        = g.GetAttr<nnvm::ShapeVector>("shape_inputs");
  const auto& dtype_inputs        = g.GetAttr<nnvm::DTypeVector>("dtype_inputs");
  const auto& storage_type_inputs = g.GetAttr<StorageTypeVector>("storage_type_inputs");

  nnvm::ShapeVector sub_shape(sub_idx.num_node_entries());
  nnvm::DTypeVector sub_dtype(sub_idx.num_node_entries());
  StorageTypeVector sub_storage_type(sub_idx.num_node_entries());
  nnvm::ShapeVector sub_shape_inputs(sub_idx.input_nodes().size());
  nnvm::DTypeVector sub_dtype_inputs(sub_idx.input_nodes().size());
  StorageTypeVector sub_storage_type_inputs(sub_idx.input_nodes().size());

  const std::unordered_map<uint32_t, uint32_t> inputsindex2pos     = GetGraphInputsMap(g);
  const std::unordered_map<uint32_t, uint32_t> sub_inputsindex2pos = GetGraphInputsMap(subgraph);
  // map attributes from graph to subgraph
  for (auto& p : old2new) {
    const uint32_t nid     = idx.node_id(p.first);
    const uint32_t sub_nid = sub_idx.node_id(p.second.get());
    const nnvm::Op* op = sub_idx[sub_nid].source->op();
    if (op == nullptr) {  // if it's an input node, there is only one output node entry
      const uint32_t sub_i       = sub_idx.entry_id(sub_nid, 0);
      const uint32_t sub_input_i = sub_inputsindex2pos.at(sub_nid);
      const uint32_t i           = idx.entry_id(nid, 0);

      sub_shape[sub_i] = shape[i];
      sub_dtype[sub_i] = dtype[i];
      sub_storage_type[sub_i]       = storage_type[i];
      sub_shape_inputs[sub_input_i] = shape_inputs[inputsindex2pos.at(nid)];
      sub_dtype_inputs[sub_input_i] = dtype_inputs[inputsindex2pos.at(nid)];
      sub_storage_type_inputs[sub_input_i] = storage_type_inputs[inputsindex2pos.at(nid)];

    } else {
      for (size_t oi = 0; oi < op->num_outputs; ++oi) {
        const uint32_t sub_i = sub_idx.entry_id(sub_nid, oi);
        const uint32_t i = idx.entry_id(nid, oi);
          sub_shape[sub_i] = shape[i];
          sub_dtype[sub_i] = dtype[i];
          sub_storage_type[sub_i] = storage_type[i];
      }
    }
  }
  // old2new doesn't contain placeholder / interfaces
  for (auto& p : main_input_entry_to_sub) {
    nnvm::NodeEntry main_entry = p.first;
    nnvm::NodeEntry sub_entry = p.second;
    const uint32_t sub_nid = sub_idx.node_id(sub_entry.node.get());
    const uint32_t sub_i = sub_idx.entry_id(sub_entry);
    const uint32_t i = idx.entry_id(main_entry);
    const uint32_t sub_input_i = sub_inputsindex2pos.at(sub_nid);
    sub_shape[sub_i] = shape[i];
    sub_dtype[sub_i] = dtype[i];
    sub_storage_type[sub_i] = storage_type[i];
    sub_shape_inputs[sub_input_i] = sub_shape[sub_i];
    sub_dtype_inputs[sub_input_i] = sub_dtype[sub_i];
    sub_storage_type_inputs[sub_input_i] = sub_storage_type[sub_i];
  }
  subgraph.attrs["shape"] =
      std::make_shared<dmlc::any>(std::move(sub_shape));
  subgraph.attrs["dtype"] =
      std::make_shared<dmlc::any>(std::move(sub_dtype));
  subgraph.attrs["storage_type"] =
      std::make_shared<dmlc::any>(std::move(sub_storage_type));
  subgraph.attrs["shape_inputs"] =
      std::make_shared<dmlc::any>(std::move(sub_shape_inputs));
  subgraph.attrs["dtype_inputs"] =
      std::make_shared<dmlc::any>(std::move(sub_dtype_inputs));
  subgraph.attrs["storage_type_inputs"] =
      std::make_shared<dmlc::any>(std::move(sub_storage_type_inputs));

  return subgraph;
}

/*!
 * \brief Generate a name for a new TRT node, avoid collision if some TRT_nodes are already defined
 */
const std::string GetNewTrtName(const Graph& g, const Graph& subgraph) {
  const std::string name_prefix("TRT_node");
  std::unordered_set<std::string> name_set;
  DFSVisit(g.outputs, [&name_set, &name_prefix](const nnvm::NodePtr& node) {
    if (node->attrs.name.compare(0, name_prefix.size(), name_prefix) == 0) {
      name_set.insert(node->attrs.name);
    }
  });
  // name inside the subgraph will be avaible as they will be removed
  DFSVisit(subgraph.outputs, [&name_set, &name_prefix](const nnvm::NodePtr& node) {
    if (node->attrs.name.compare(0, name_prefix.size(), name_prefix) == 0) {
      name_set.erase(node->attrs.name);
    }
  });
  uint32_t name_suffix = 0;
  std::string full_name = name_prefix + std::to_string(name_suffix);
  while (name_set.count(full_name)) {
    full_name = name_prefix + std::to_string(++name_suffix);
  }
  return full_name;
}

/*!
 * \brief helper function to display what nodes are in a specific subset
 */
void dispNodesSet(Graph g, std::unordered_set<nnvm::Node*> s) {
  DFSVisit(g.outputs, [&s](const nnvm::NodePtr n){
    if (s.count(n.get())) {
      std::cout << "  Y " << n->attrs.name << std::endl;
    } else {
      std::cout << "  N " << n->attrs.name << std::endl;
    }
  });
}

/*!
 * \brief Replace a set of nodes by a TensorRT node
 */
Graph ReplaceSubgraph(Graph&& g,
                      const std::unordered_set<nnvm::Node*>& set_subgraph,
                      std::unordered_map<std::string, NDArray>* const params_map) {
  // Create MXNet subgraph
  Graph subgraph;

  const auto sub_outputs_in_main = GetSubgraphNodeEntries(g, set_subgraph);
  subgraph.outputs = sub_outputs_in_main;
  // old2new will link raw pointer of the nodes in the graph to
  // the corresponding shared_ptr of the nodes in the generated subgraph
  std::unordered_map<nnvm::Node*, nnvm::NodePtr> old2new;
  std::deque<nnvm::Node*> stack;
  std::unordered_set<nnvm::Node*> visited;
  int32_t reservation = set_subgraph.size();
  old2new.reserve(reservation);
  visited.reserve(reservation);

  // Create the shared_ptr using the same raw pointer don't really matter
  for (auto& n : set_subgraph) {
    old2new[n] = std::make_shared<nnvm::Node>(*n);
  }

  // To generate a subgraph an input have to be replace by data node (no op)
  // and it have to be agnostic to the node from which it's an output
  // (For exemple even if two inputs are two different outputs from the same node)
  nnvm::NodeEntryMap<nnvm::NodeEntry> main_input_entry_to_sub;
  for (auto& e : GetSubgraphInterfaceNodes(g, set_subgraph)) {
    auto node = nnvm::Node::Create();
    node->attrs.name = e.node->attrs.name + "_" + std::to_string(e.index);
    auto new_e = nnvm::NodeEntry{node, 0, 0};
    main_input_entry_to_sub[e] = new_e;
  }

  for (nnvm::NodeEntry& e : subgraph.outputs) {
    e.node = old2new[e.node.get()];
    stack.emplace_back(e.node.get());
  }
  // link all nodes in the subgraph to nodes in the subgraph instead of main graph
  while (!stack.empty()) {
    auto vertex = stack.front();
    stack.pop_front();
    if (!visited.count(vertex)) {
      visited.insert(vertex);
      for (auto& e : vertex->inputs) {
        auto it = main_input_entry_to_sub.find(e);
        if (it != main_input_entry_to_sub.end()) {
          e = it->second;
        } else {
          e.node = old2new[e.node.get()];
        }
      stack.emplace_back(e.node.get());
      }
    }
  }
  // Remove the control dependencies of the subgraph to nodes that are not in the subgraph
  DFSVisit(subgraph.outputs, [&set_subgraph, &old2new](const nnvm::NodePtr& node) {
    std::remove_if(node->control_deps.begin(),
                   node->control_deps.end(),
                   [&set_subgraph](nnvm::NodePtr n_ptr) {
                    return !set_subgraph.count(n_ptr.get());
                   });
    for (nnvm::NodePtr& n_ptr : node->control_deps) {
      n_ptr = old2new[n_ptr.get()];
    }
  });

  subgraph = UpdateSubgraphAttrs(std::move(subgraph), g, old2new, main_input_entry_to_sub);
  auto& sub_idx = subgraph.indexed_graph();

  auto trtnodeptr = ConvertNnvmGraphToOnnx(subgraph, params_map);
  trtnodeptr->attrs.name = GetNewTrtName(g, subgraph);

  // Insert new trt node and unplug replaced nodes
  std::unordered_map<uint32_t, nnvm::NodeEntry> sub_input_entryid_to_main;
  for (auto& p : main_input_entry_to_sub) {
    sub_input_entryid_to_main[sub_idx.entry_id(p.second)] = p.first;
  }

  // Plug the nodes from the main graph as inputs of the trt node
  trtnodeptr->inputs.resize(main_input_entry_to_sub.size());
  {
    uint32_t counter = 0;
    for (uint32_t i : sub_idx.input_nodes()) {
      auto it = sub_input_entryid_to_main.find(sub_idx.entry_id(i, 0));
      if (it != sub_input_entryid_to_main.end()) {
        trtnodeptr->inputs[counter++] = it->second;
      }
    }
  }
  nnvm::NodeEntryMap<uint32_t> sub_outputs_in_main_to_pos;
  for (uint32_t i = 0; i < sub_outputs_in_main.size(); ++i) {
    sub_outputs_in_main_to_pos[sub_outputs_in_main[i]] = i;
  }
  // Plug the trt node as inputs to the main graph nodes
  DFSVisit(g.outputs, [&sub_outputs_in_main_to_pos, &trtnodeptr](const nnvm::NodePtr& n) {
    for (auto& e : n->inputs) {
      auto it = sub_outputs_in_main_to_pos.find(e);
      if (it != sub_outputs_in_main_to_pos.end()) {
        e.index = it->second;
        e.node = trtnodeptr;
      }
    }
  });

  for (auto& output : g.outputs) {
    auto it = sub_outputs_in_main_to_pos.find(output);
    if (it != sub_outputs_in_main_to_pos.end()) {
      output.index = it->second;
      output.node = trtnodeptr;
    }
  }

  Graph new_graph;
  new_graph.outputs = g.outputs;
  return new_graph;
}

std::vector<std::unordered_set<nnvm::Node*>> GetTrtCompatibleSubsets(const Graph& g,
    std::unordered_map<std::string, NDArray>* const params_map) {
  BidirectionalGraph biG = BidirectionalGraph(g);
  std::vector<std::unordered_set<BidirectionalGraph::Node*>> subsets = biG.get_subsets(params_map);
  std::vector<std::unordered_set<nnvm::Node*>> nnvm_subsets(subsets.size(),
                                                            std::unordered_set<nnvm::Node*>());
  for (size_t i = 0; i < subsets.size(); ++i) {
    nnvm_subsets[i].reserve(subsets[i].size());
    for (auto& n : subsets[i]) {
      nnvm_subsets[i].insert(n->nnvmptr);
    }
  }
  return nnvm_subsets;
}

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
