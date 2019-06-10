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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_fc_property.cc
 * \brief Partition gragph property for FullyConnected operator
 * \author Ciyong Chen
*/

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "../subgraph_property.h"

namespace mxnet {
namespace op {

class SgMKLDNNFCSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  bool disable_fc_relu;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;

 public:
  explicit SgMKLDNNFCSelector(const bool dis_fc_relu) : disable_fc_relu(dis_fc_relu) {}

  bool Select(const nnvm::Node &n) override {
    if (n.op() == Op::Get("FullyConnected")) {
      status = disable_fc_relu ? kSuccess : kStart;
      matched_list.clear();
      matched_list.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (status == kFail || status == kSuccess || new_node.is_variable())
      return false;

    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list.back() != &n) {
      if (std::find(matched_list.begin(), matched_list.end(), &n) !=
        matched_list.end()) {
        while (matched_list.back() != &n) {
          matched_list.pop_back();
        }
      }

      status = kSuccess;
      return false;
    }

    switch (status) {
      case kStart:
        if (new_node.op() == Op::Get("Activation") &&
            new_node.attrs.dict.at("act_type") == "relu") {
          matched_list.push_back(&new_node);
          status = kSuccess;
          return true;
        }
      default:
        status = kSuccess;
        return false;
    }
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status == kFail) {
      return std::vector<nnvm::Node *>(0);
    } else {
      std::vector<nnvm::Node *> ret;
      for (auto i : matched_list) {
        auto non_const_i = const_cast<nnvm::Node *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return candidates;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list.size(), 1);
    auto new_selector = SgMKLDNNFCSelector(disable_fc_relu);
    new_selector.Select(*matched_list[0]);
    *this = new_selector;
  }
};

class SgMKLDNNFCProperty : public SubgraphProperty {
 public:
  SgMKLDNNFCProperty() {
    disable_fc_relu = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FUSE_FC_RELU", false);
  }

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN FullyConnected optimization pass";
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FC_OPT", 0)) {
      LOG(INFO) << name << " is disabled.";
      return nullptr;
    }
    auto property = std::make_shared<SgMKLDNNFCProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    node_name << "sg_mkldnn_";
    DFSVisit(new_sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &sub_name = node->op()->name;
      if (sub_name == "FullyConnected") {
        node_name << "fully_connected_";
      } else if ((sub_name == "Activation") &&
                 (node->attrs.dict.at("act_type") == "relu")) {
          node_name << "relu_";
          n->attrs.dict["with_relu"] = "True";
      }
    });
    node_name << std::to_string(subgraph_id);
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_fully_connected");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNFCSelector>(disable_fc_relu);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

 private:
  bool disable_fc_relu;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
