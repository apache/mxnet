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

#ifndef MXNET_OPERATOR_SUBGRAPH_DEFAULT_SUBGRAPH_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DEFAULT_SUBGRAPH_PROPERTY_H_

#include <vector>
#include <string>
#include "./common.h"
#include "./subgraph_property.h"

namespace mxnet {
namespace op {

/*
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class ContainOpSelector: public SubgraphSelector {
 public:
  explicit ContainOpSelector(const std::unordered_set<std::string>& op_names)
    : op_names_(op_names) {}

  virtual bool Select(const nnvm::Node &n) {
    return !n.is_variable() && op_names_.count(n.op()->name);
  }

  virtual bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) {
    return !new_node.is_variable() && op_names_.count(new_node.op()->name);
  }

  virtual bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) {
    return !new_node.is_variable() && op_names_.count(new_node.op()->name);
  }
 private:
  const std::unordered_set<std::string>& op_names_;
};

/*
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _default_subgraph_op.
 */
class DefaultSubgraphProperty: public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() { return std::make_shared<DefaultSubgraphProperty>(); }
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                           const int subgraph_id = 0) const {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_default_subgraph_op");
    n->attrs.name = "_default_subgraph_op" + std::to_string(subgraph_id);
    n->attrs.parsed = sym;
    return n;
  }
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<ContainOpSelector>(
        this->GetAttr<std::unordered_set<std::string>>("op_names"));
  }
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(default, DefaultSubgraphProperty)

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_DEFAULT_SUBGRAPH_PROPERTY_H_
