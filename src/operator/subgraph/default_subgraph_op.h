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

#ifndef MXNET_OPERATOR_SUBGRAPH_DEFAULT_SUBGRAPH_OP_H_
#define MXNET_OPERATOR_SUBGRAPH_DEFAULT_SUBGRAPH_OP_H_

#include <mxnet/graph_attr_types.h>
#include <string>
#include <vector>
#include "./common.h"

namespace mxnet {
namespace op {

/*
 * This provides criteria for selecting nodes in a subgraph.
 * When a node is passed to this object, the selection criteria may be changed.
 * We can also specify what links we should use when traversing the neighbor
 * nodes.
 */
class SubgraphSelector {
 public:
  virtual ~SubgraphSelector() {}
  // Determine if the node should be selected for a subgraph.
  virtual bool Select(const nnvm::Graph &g, const nnvm::Node &n) = 0;
  // Determine if the input node should be selected for a subgraph.
  virtual bool SelectInput(const nnvm::Graph &g, const nnvm::Node &n,
                           const nnvm::Node &new_node) = 0;
  // Determine if the output node should be selected for a subgraph.
  virtual bool SelectOutput(const nnvm::Graph &g, const nnvm::Node &n,
                            const nnvm::Node &new_node) = 0;
  // Post processes pre-selected subgraph nodes. Return a list of nodes that
  // users want to keep in subgraph(s).
  virtual std::vector<nnvm::Node *> Filter(const nnvm::Graph &g,
                                           const std::vector<nnvm::Node *> &candidates) {
    return candidates;
  }
};

using SubgraphSelectorPtr = std::shared_ptr<SubgraphSelector>;

/*!
 * \brief This provides a set of properties for partitioning a graph into subgraphs,
 * reconstructing a new graph from the subgraphs and creating a subgraph
 * operator to execute the subgraph.
 */
class SubgraphProperty {
 public:
  // the criteria of selecting the subgraph nodes.
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const = 0;
  // create an nnvm node for a given subgraph. Here users can customize how to
  // execute the operators in the subgraph.
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &s,
                                           const int subgraph_id = 0) const = 0;
};

using SubgraphPropertyPtr = std::shared_ptr<SubgraphProperty>;

void RegisterSubgraphProperty(SubgraphPropertyPtr property);

/*
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class ContainOpSelector : public SubgraphSelector {
  std::shared_ptr<const std::unordered_set<std::string>> op_names;

 public:
  explicit ContainOpSelector(std::shared_ptr<const std::unordered_set<std::string>> op_names) {
    this->op_names = op_names;
  }

  virtual bool Select(const nnvm::Graph &g, const nnvm::Node &n) {
    return !n.is_variable() && op_names->count(n.op()->name);
  }

  virtual bool SelectInput(const nnvm::Graph &g, const nnvm::Node &n, const nnvm::Node &new_node) {
    return !new_node.is_variable() && op_names->count(new_node.op()->name);
  }

  virtual bool SelectOutput(const nnvm::Graph &g, const nnvm::Node &n,
                            const nnvm::Node &new_node) {
    return !new_node.is_variable() && op_names->count(new_node.op()->name);
  }
};

/*
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _default_subgraph_op.
 */
class DefaultSubgraphProperty : public SubgraphProperty {
 public:
  explicit DefaultSubgraphProperty(
      const std::unordered_set<std::string> &op_names = std::unordered_set<std::string>{})
      : op_names_(std::make_shared<std::unordered_set<std::string>>(op_names)) {}
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                           const int subgraph_id = 0) const {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_default_subgraph_op");
    n->attrs.name = "_default_subgraph_op" + std::to_string(subgraph_id);
    n->attrs.parsed = sym;
    return n;
  }
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<ContainOpSelector>(op_names_);
  }

 private:
  std::shared_ptr<const std::unordered_set<std::string>> op_names_;
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_DEFAULT_SUBGRAPH_OP_H_
