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

#include "./common.h"
#include "./subgraph_property.h"
#include "../../imperative/cached_op.h"

namespace mxnet {
namespace op {

/*
 * This selects nodes for a subgraph that only contains static shape operators
 * and it visits nodes via both input and output links.
 */
class StaticShapeOpSelector: public SubgraphSelector {
 public:
  // select nodes with FInferShape attribute
  bool Select(const nnvm::Node &seed_node) override {
    const auto& infershape = nnvm::Op::GetAttr<mxnet::FInferShape>("FInferShape");
    return !seed_node.is_variable() && infershape.count(seed_node.op()) &&
           !unsupported_op_names_.count(seed_node.op()->name);
  }

  bool SelectInput(const nnvm::Node &cur_node, const nnvm::Node &input_node) override {
    return Select(input_node);
  }

  bool SelectOutput(const nnvm::Node &cur_node, const nnvm::Node &output_node) override {
    return Select(output_node);
  }

  // reject single node subgraph
  std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) override {
    if (candidates.size() == 1) {
      return std::vector<nnvm::Node*>();
    }
    return candidates;
  }

 private:
    // static shape ops that fail backward pass inside subgraph CachedOp
    // GitHub issue: https://github.com/apache/incubator-mxnet/issues/18823
    std::unordered_set<std::string> unsupported_op_names_ {"Reshape", "_np_reshape", "transpose",
                                                           "_npi_dstack", "_npi_hstack"};
};

/*
 * This subgraph property finds a subgraph whose nodes have only static shape operators.
 * The operators in the subgraph will be executed by _CachedOp.
 */
class StaticShapeSubgraphProperty: public SubgraphProperty {
 public:
  StaticShapeSubgraphProperty() {
    // flag to recursively partition dynamic shape op nodes containing subgraphs
    attrs_["recursive_partition"] = std::make_shared<dmlc::any>(true);
    // flag to ensure subgraph CachedOp has at least one external input
    // as required by CachedOp::Forward
    attrs_["ensure_CachedOp_input"] = std::make_shared<dmlc::any>(true);
  }
  static SubgraphPropertyPtr Create() { return std::make_shared<StaticShapeSubgraphProperty>(); }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    return std::make_shared<StaticShapeOpSelector>();
  }

  void PrePartition(const nnvm::Graph& g,
                    const std::unordered_map<std::string, std::string>& options_map) override {
    for (auto& kv : options_map) {
      options_map_.emplace_back(kv);
    }
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_CachedOp");
    n->attrs.name = "_static_shape_CachedOp" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    n->attrs.parsed = std::make_shared<CachedOp>(sym, options_map_);
    return n;
  }

 private:
  std::vector<std::pair<std::string, std::string>> options_map_;
};

MXNET_REGISTER_SUBGRAPH_BACKEND(static_shape);
MXNET_REGISTER_SUBGRAPH_PROPERTY(static_shape, StaticShapeSubgraphProperty);

}  // namespace op
}  // namespace mxnet

