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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_BN_RELU_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_BN_RELU_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "mkldnn_subgraph_base-inl.h"
#include "../../nn/mkldnn/mkldnn_batch_norm-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNBNReLUSelector : public SubgraphSelector {
 public:
  explicit SgMKLDNNBNReLUSelector(const bool disable_bn_relu) :
      disable_bn_relu_(disable_bn_relu), relu_after_bn_(false) {}

  bool Select(const nnvm::Node &n) override {
    return n.op() && n.op()->name == "BatchNorm";
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (new_node.op() && new_node.op()->name == "Activation" && !relu_after_bn_ &&
        nnvm::get<ActivationParam>(new_node.attrs.parsed).act_type == activation::kReLU) {
      relu_after_bn_ = true;
      return true;
    }
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (!disable_bn_relu_ && relu_after_bn_)
      return candidates;
    else
      return std::vector<nnvm::Node *>();
  }

 private:
  bool disable_bn_relu_;
  bool relu_after_bn_;
};

class SgMKLDNNBNReLUProperty : public SubgraphProperty {
 public:
  SgMKLDNNBNReLUProperty() {
    disable_bn_relu_ = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FUSE_BN_RELU", false);
  }

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN BN + ReLU optimization pass";
    auto property = std::make_shared<SgMKLDNNBNReLUProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_BN_RELU_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();

    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    node_name << "sg_mkldnn_batch_norm_relu_" << std::to_string(subgraph_id);

    BatchNormParam param;
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if (!node->is_variable() && node->op()->name == "BatchNorm") {
        param = nnvm::get<BatchNormParam>(node->attrs.parsed);
      }
    });

    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_contrib_BatchNormWithReLU");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->attrs.parsed = param;
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNBNReLUSelector>(disable_bn_relu_);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::ObjectPtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

 private:
  bool disable_bn_relu_;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_BN_RELU_PROPERTY_H_
