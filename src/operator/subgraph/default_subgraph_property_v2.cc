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

#include <memory>

#include "./common.h"
#include "./subgraph_property.h"
#include "../../imperative/cached_op.h"

namespace mxnet {
namespace op {

/*!
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class ContainOpSelectorV2 : public SubgraphSelectorV2 {
 public:
  explicit ContainOpSelectorV2(const std::unordered_set<std::string>& op_names)
      : op_names_(op_names) {}

  bool Select(const BiDirectedNode& sn) override {
    const auto& seed_node = *sn.node;
    return !seed_node.is_variable() && op_names_.count(seed_node.op()->name);
  }

  bool SelectInput(const BiDirectedNode& sn, const BiDirectedNode& snew_node) override {
    const auto& input_node = *snew_node.node;
    return !input_node.is_variable() && op_names_.count(input_node.op()->name);
  }

  bool SelectOutput(const BiDirectedNode& sn, const BiDirectedNode& snew_node) override {
    const auto& output_node = *snew_node.node;
    return !output_node.is_variable() && op_names_.count(output_node.op()->name);
  }

 private:
  const std::unordered_set<std::string>& op_names_;
};

/*!
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _CachedOp.
 */
class DefaultSubgraphProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    return std::make_shared<DefaultSubgraphProperty>();
  }
  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const SubgraphSelectorPtr& subgraph_selector,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs.op       = Op::Get("_CachedOp");
    n->attrs.name     = "_CachedOp" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));

    std::vector<std::pair<std::string, std::string>> flags{{"static_alloc", "true"}};
    n->attrs.parsed = std::make_shared<CachedOp>(sym, flags);

    return n;
  }
  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    return std::make_shared<ContainOpSelectorV2>(
        this->GetAttr<std::unordered_set<std::string>>("op_names"));
  }
};

MXNET_REGISTER_SUBGRAPH_BACKEND(default_v2);

MXNET_REGISTER_SUBGRAPH_PROPERTY(default_v2, DefaultSubgraphProperty);

}  // namespace op
}  // namespace mxnet
