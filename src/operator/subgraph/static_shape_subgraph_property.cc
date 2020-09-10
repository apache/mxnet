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
    // update static_alloc and static_shape flags
    if (options_map_.size() == 0) {
      for (auto& kv : options_map) {
        options_map_.emplace_back(kv);
      }
    }
    if (param_name_set_.size() > 0) {
      param_name_set_.clear();
    }
    // generate param_name_set_ for data_indices and param_indices
    if (g.HasAttr("param_indices_list")) {
      const std::vector<int>& param_indices_list
          = g.GetAttr<std::vector<int>>("param_indices_list");
      const auto& indexed_graph = g.indexed_graph();
      for (const auto index : param_indices_list) {
          auto nid = indexed_graph.input_nodes()[index];
          nnvm::Node n = *(indexed_graph[nid].source);
          param_name_set_.emplace(n.attrs.name);
      }
    }
  }

  void InitSubgraphInputs(std::vector<nnvm::NodeEntry*>* input_entries,
                          std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    for (size_t i = 0; i < input_entries->size(); ++i) {
      nnvm::NodeEntry *e = input_entries->at(i);
      nnvm::NodeEntry& orig = orig_input_entries->at(i);
      // set attribute for subgraph input nodes
      if (orig.node->is_variable()) {
        // get name of original output entry
        nnvm::Symbol sym;
        sym.outputs.push_back(orig);
        const auto output_names = sym.ListOutputNames();
        CHECK_EQ(output_names.size(), 1U);
        const std::string& var_name = output_names[0];
        e->node->attrs.dict["isArg"] = "True";
        e->node->attrs.dict["argName"] = var_name;
      } else {
        e->node->attrs.dict["isArg"] = "False";
      }
    }
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                     const int subgraph_id = 0) const override {
    std::vector<std::pair<std::string, std::string>> flags;
    _set_cachedop_flags(sym, &flags);
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_CachedOp");
    n->attrs.name = "_static_shape_CachedOp" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    n->attrs.parsed = std::make_shared<CachedOp>(sym, flags);
    return n;
  }

 private:
    // generate data_indices and param_indices for subgraph CachedOp node
    void _set_cachedop_flags(nnvm::Symbol symbol,
             std::vector<std::pair<std::string, std::string>>* flags) const {
    std::vector<nnvm::ObjectPtr> inputs = symbol.ListInputs(nnvm::Symbol::ListInputOption(0));
    std::string data_indices = "[";
    std::string param_indices = "[";
    for (int i = 0; i < inputs.size(); i++) {
      nnvm::ObjectPtr node = inputs[i];
      if (node->attrs.dict["isArg"] == "True"
          && param_name_set_.count(node->attrs.dict["argName"]) > 0) {
        if (param_indices.compare("[") == 0) {
          param_indices = param_indices + std::to_string(i);
        } else {
          param_indices = param_indices + ", " + std::to_string(i);
        }
      } else {
        if (data_indices.compare("[") == 0) {
          data_indices = data_indices + std::to_string(i);
        } else {
          data_indices = data_indices + ", " + std::to_string(i);
        }
      }
    }
    data_indices = data_indices + "]";
    param_indices = param_indices + "]";
    for (auto kv : options_map_) {
      flags->emplace_back(kv);
    }
    flags->emplace_back("data_indices", data_indices);
    flags->emplace_back("param_indices", param_indices);
  }

  std::vector<std::pair<std::string, std::string>> options_map_;
  std::set<std::string> param_name_set_;
};

MXNET_REGISTER_SUBGRAPH_BACKEND(static_shape);
MXNET_REGISTER_SUBGRAPH_PROPERTY(static_shape, StaticShapeSubgraphProperty);

}  // namespace op
}  // namespace mxnet

