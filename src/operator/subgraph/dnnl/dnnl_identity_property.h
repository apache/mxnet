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
 * \file dnnl_identity_property.cc
 * \brief Graph property for removing identity operators
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_IDENTITY_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_IDENTITY_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <map>
#include <string>
#include <vector>

#include "../common.h"
#include "../../nn/dropout-inl.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgDNNLIdentitySelector : public SubgraphSelectorV2 {
 private:
  std::vector<const BiDirectedNode*> matched_list_;
  bool pattern_found = false;

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    bool status = false;
    if (seed_node.node->op() == Op::Get("_npi_copy")) {
      status = true;
    }

    if (seed_node.node->op() == Op::Get("Dropout")) {
      auto const& dropout_param = nnvm::get<DropoutParam>(seed_node.node->attrs.parsed);
      if (dropout_param.mode == dropout::kTraining) {
        status = true;
      }
    }

    if (status) {
      matched_list_.clear();
      matched_list_.emplace_back(&seed_node);
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    if (pattern_found || input_node.node->is_variable()) {
      return false;
    } else if (input_node.node->op()) {
      matched_list_.emplace_back(&input_node);
      pattern_found = true;
      return true;
    }
    return false;
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    return false;
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    // candidates should contain only two nodes - custom node and identity node
    if (pattern_found && candidates.size() == matched_list_.size()) {
      CHECK_EQ(candidates.size(), 2);
      return candidates;
    } else {
      return std::vector<BiDirectedNode*>(0);
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgDNNLIdentitySelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

inline bool IsIdentityNode(const nnvm::ObjectPtr node) {
  return node->op() && (node->op() == Op::Get("_npi_copy") || node->op() == Op::Get("Dropout"));
}

class SgDNNLIdentityProperty : public SubgraphProperty {
 public:
  SgDNNLIdentityProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "DNNL Identity optimization pass";
    auto property                  = std::make_shared<SgDNNLIdentityProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::NodeEntry identity_node_entry;
    for (auto entry : sym.outputs) {
      if (IsIdentityNode(entry.node)) {
        identity_node_entry = entry;
      }
    }

    auto last_node = identity_node_entry.node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);

    nnvm::ObjectPtr org_node;
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (!IsIdentityNode(node)) {
        org_node = node;
      }
    });

    // Create copy of original node
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs          = org_node->attrs;
    if (n->op()->attr_parser) {
      n->op()->attr_parser(&(n->attrs));
    }
    return n;
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    // output of identity must be connected as output of operator before identity
    // e.g. for:        /--index 0--> custom_op
    //         (n) slice
    //                  \--index 1--> Dropout --index 0--> OUT_NODE
    //  for OUT_NODE index 0 must be changed to index 1
    for (int i = 0; i < output_entries->size(); ++i) {
      auto out_node = output_entries->at(i)->node;
      if (IsIdentityNode(out_node)) {
        output_entries->at(i)->index = out_node->inputs[0].index;
      }
      output_entries->at(i)->node = n;
    }
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLIdentitySelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_IDENTITY_PROPERTY_H_
