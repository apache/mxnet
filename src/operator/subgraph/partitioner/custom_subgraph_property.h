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

/*
 * This file contains an implementation of a subgraph property
 * that interfaces between MXNet and custom subgraph properties
 * created by users in external libraries. It does not implement
 * any custom subgraphing logic itself, rather it calls APIs
 * in the user's custom library to enable control of partitioning
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_

#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include <string>
#include <utility>
#include <vector>
#include "../common.h"
#include "../subgraph_property.h"
#include "../../include/mxnet/lib_api.h"
namespace mxnet {
namespace op {

/*
 * This selects nodes for a subgraph based on node name as supplied
 * by the supportedOps from an external library. It visits nodes via
 * both input and output links.
 */
class CustomContainOpSelector: public SubgraphSelector {
 public:
  explicit CustomContainOpSelector(std::unordered_set<std::string> supportedNodes) :
    supportedNodes_(supportedNodes) {}
  virtual bool Select(const nnvm::Node &n) {
    return supportedNodes_.count(n.attrs.name) > 0;
  }
  virtual bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) {
    return supportedNodes_.count(new_node.attrs.name) > 0;
  }
  virtual bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) {
    return supportedNodes_.count(new_node.attrs.name) > 0;
  }
  std::unordered_set<std::string> supportedNodes_;
};

/*
 * This subgraph property finds a subgraph that only contains
 * nodes as specified by the supportedOps from an external library.
 * The operators in the subgraph will be executed by the operator
 * specified by the external library too.
 */
class  CustomSubgraphProperty: public SubgraphProperty {
 public:
  CustomSubgraphProperty() :
    subgraphProp("error"),
    callSupportedOps_(nullptr),
    supportedOps_(nullptr),
    callAcceptSubgraph_(nullptr),
    acceptSubgraph_(nullptr),
    subgraph_op_name("error") {}
  CustomSubgraphProperty(std::string subgraphProp_name,
                         partCallSupportedOps_t callSupportedOps,
                         supportedOps_t supportedOps,
                         partCallAcceptSubgraph_t callAcceptSubgraph,
                         acceptSubgraph_t acceptSubgraph,
                         std::string op_name) :
    subgraphProp(subgraphProp_name),
      callSupportedOps_(callSupportedOps),
      supportedOps_(supportedOps),
      callAcceptSubgraph_(callAcceptSubgraph),
      acceptSubgraph_(acceptSubgraph),
      subgraph_op_name(op_name) {}

  // create custom subgraph property
  static SubgraphPropertyPtr Create() {
    return std::make_shared<CustomSubgraphProperty>();
  }

  void PrePartition(const nnvm::Graph& g,
    const std::vector<std::pair<std::string, std::string>>& options_map) {
    // clear supportedNodes to remove state from previous calls
    supportedNodes.clear();

    // remove all graph attrs, some cannot be saved to json
    nnvm::Graph graph = std::move(g);
    graph.attrs.clear();
    const nnvm::IndexedGraph& indexed_graph = graph.indexed_graph();

    // set shape attrs for each node in the graph
    if (g.HasAttr("shape")) {
      mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
      for (unsigned i = 0; i < indexed_graph.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[i].source);
        mxnet::TShape shape = shapes[i];
        std::stringstream ss;
        ss << shape;
        node->attrs.dict["shape"] = ss.str();
      }
    }
    // set dtype attrs for each node in the graph
    if (g.HasAttr("dtype")) {
      std::vector<int> dtypes = g.GetAttr<std::vector<int> >("dtype");
      for (unsigned i = 0; i < indexed_graph.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[i].source);
        int dtype = dtypes[i];
        std::stringstream ss;
        ss << dtype;
        node->attrs.dict["dtype"] = ss.str();
      }
    }

    CHECK(supportedOps_ != nullptr)
      << "supportedOps_ is null for " << subgraphProp << std::endl;
    CHECK(callSupportedOps_ != nullptr)
      << "callSupportedOps_ is null for " << subgraphProp << std::endl;

    std::string subgraph_json = nnvm::pass::SaveJSON(graph);
    std::vector<int> supportedNodeIDs(indexed_graph.num_nodes(), 0);
    const char* json = subgraph_json.c_str();
    int *ids = supportedNodeIDs.data();

    // clear options from previous call
    opt_keys_.clear();
    opt_vals_.clear();
    options_map_.clear();
    for (auto kv : options_map) {
      options_map_.push_back(kv);
      opt_keys_.push_back(options_map_.back().first.c_str());
      opt_vals_.push_back(options_map_.back().second.c_str());
    }

    CHECK(callSupportedOps_(supportedOps_, json, supportedNodeIDs.size(), ids,
                            opt_keys_.data(), opt_vals_.data(), opt_keys_.size()))
      << "Error calling supportedOps for '" << subgraphProp << "'";

    const auto& idx = g.indexed_graph();
    // loop and add node names for each supported node ID
    for (unsigned i = 0; i < supportedNodeIDs.size(); i++) {
      if (supportedNodeIDs[i]) {
        supportedNodes.insert(idx[i].source->attrs.name);
      }
    }
  }
  // override CreateSubgraphNode
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                           const int subgraph_id = 0) const {
    int accept = 1;
    if (acceptSubgraph_) {
      nnvm::Graph g;
      g.outputs = sym.outputs;
      const auto& idx = g.indexed_graph();

      // set isArg/isAux for each null op/param in the graph
      const std::vector<std::string> aux_names = sym.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      std::unordered_set<std::string> aux_set(aux_names.begin(), aux_names.end());
      for (unsigned i = 0; i < idx.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(idx[i].source);
        // check if this node is input to subgraph
        if (node->is_variable()) {
          // check if this node is an aux param
          if (aux_set.count(node->attrs.name))
            node->attrs.dict["isAux"] = "True";
          else
            node->attrs.dict["isAux"] = "False";
        }
      }

      std::string subgraph_json = nnvm::pass::SaveJSON(g);
      CHECK(callAcceptSubgraph_(acceptSubgraph_, subgraph_json.c_str(),
                                subgraph_id, &accept, opt_keys_.data(),
                                opt_vals_.data(), opt_keys_.size()))
        << "Error calling acceptSubgraph for '" << subgraphProp << "'";
    }
    if (accept) {
      nnvm::NodePtr n = nnvm::Node::Create();
      n->attrs.op = Op::Get(subgraph_op_name);
      n->attrs.name = "_op" + std::to_string(subgraph_id);
      n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
      return n;
    } else {
      return NULL;
    }
  }
  // override CreateSubgraphSelector
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<CustomContainOpSelector>(supportedNodes);
  }

  std::string subgraphProp;
  partCallSupportedOps_t callSupportedOps_;
  supportedOps_t supportedOps_;
  partCallAcceptSubgraph_t callAcceptSubgraph_;
  acceptSubgraph_t acceptSubgraph_;
  std::unordered_set<std::string> supportedNodes;
  std::string subgraph_op_name;
  std::vector<std::pair<std::string, std::string>> options_map_;
  std::vector<const char*> opt_keys_, opt_vals_;
};
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_
