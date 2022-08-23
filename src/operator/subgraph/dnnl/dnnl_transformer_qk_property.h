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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>

#include "dnnl_transformer_qk_common.h"

/*
       custom_op      custom_op
          |              |
    ______|______________|________
   |      |              |        |
   |  _npx_reshape  _npx_reshape  |
   |      |              |        |
   |  SwapAxis        SwapAxis    |
   |       \          /           |
   |        batch_dot             |
   |            |                 |
   |______________________________|

OR

              custom_op
                 |
    _____________|_________________
   |            Split             |
   |           /     \            |
   |  _npx_reshape  _npx_reshape  |
   |      |              |        |
   |  SwapAxis        SwapAxis    |
   |       \          /           |
   |        batch_dot             |
   |            |                 |
   |______________________________|

*/
namespace mxnet {
namespace op {

class SgDNNLTransformerQKSelector : public SubgraphSelectorV2 {
 private:
  qk_common::SelectStatusTransformerQK status_;
  std::vector<const BiDirectedNode*> matched_list_;

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    return qk_common::Select(&status_, &matched_list_, seed_node, node_attr);
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    return qk_common::SelectInput<false>(&status_, &matched_list_, n, input_node);
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    return false;
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    return qk_common::Filter(status_, matched_list_, candidates);
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgDNNLTransformerQKSelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgDNNLTransformerQKSplitSelector : public SubgraphSelectorV2 {
 private:
  qk_common::SelectStatusTransformerQK status_;
  std::vector<const BiDirectedNode*> matched_list_;

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    return qk_common::Select(&status_, &matched_list_, seed_node, node_attr);
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    return qk_common::SelectInput<true>(&status_, &matched_list_, n, input_node);
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    return false;
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    return qk_common::Filter(status_, matched_list_, candidates);
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgDNNLTransformerQKSplitSelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgDNNLTransformerQKProperty : public SubgraphProperty {
 public:
  SgDNNLTransformerQKProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN Transformer optimization pass";
    auto property                  = std::make_shared<SgDNNLTransformerQKProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_TRANSFORMER_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    return qk_common::CreateSubgraphNode<false>(sym, subgraph_id);
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    qk_common::ConnectSubgraphOutputs(n, output_entries);
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLTransformerQKSelector>();
    return selector;
  }
};

class SgDNNLTransformerQKSplitProperty : public SubgraphProperty {
 public:
  SgDNNLTransformerQKSplitProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN Transformer optimization pass";
    auto property                  = std::make_shared<SgDNNLTransformerQKSplitProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_TRANSFORMER_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    return qk_common::CreateSubgraphNode<true>(sym, subgraph_id);
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    qk_common::ConnectSubgraphOutputs(n, output_entries);
  }

  void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    subgraph_node->inputs.resize(1);
    // split is not part of subgraph, skip split as input and
    // connect subgraph input with split input
    subgraph_node->inputs[0] = orig_input_entries->at(0).node->inputs[0];
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLTransformerQKSplitSelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_PROPERTY_H_
