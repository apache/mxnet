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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_BN_RELU_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_BN_RELU_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>

#include "dnnl_subgraph_base-inl.h"
#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/nn/dnnl/dnnl_batch_norm-inl.h"
#include "operator/subgraph/common.h"

namespace mxnet {
namespace op {

class SgDNNLBNReLUSelector : public SubgraphSelector {
 public:
  explicit SgDNNLBNReLUSelector(const bool disable_bn_relu)
      : disable_bn_relu_(disable_bn_relu), status_(kStart) {}

  bool Select(const nnvm::Node& n) override {
    return n.op() && n.op()->name == "BatchNorm";
  }

  bool SelectInput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    if (n.op() && n.op()->name == "BatchNorm") {
      if (new_node.op() && status_ == kStart &&
          (new_node.op()->name == "relu" ||
           (new_node.op()->name == "Activation" &&
            nnvm::get<ActivationParam>(new_node.attrs.parsed).act_type == activation::kReLU))) {
        status_ = kSuccess;
        return true;
      } else {
        // Do not fuse if BatchNorm is connected to other nodes
        // e.g: ->- BN --- ReLU --- elementwise_add ->-
        //           \                   /
        //            \-------->--------/
        status_ = kFail;
        return false;
      }
    }
    return false;
  }

  std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) override {
    if (!disable_bn_relu_ && status_ == kSuccess)
      return candidates;
    else
      return std::vector<nnvm::Node*>();
  }

 private:
  bool disable_bn_relu_;
  SelectStatus status_;
};

class SgDNNLBNReLUProperty : public SubgraphProperty {
 public:
  SgDNNLBNReLUProperty() {
    disable_bn_relu_ = dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FUSE_BN_RELU", false);
  }

  void PrePartition(const nnvm::Graph& g,
                    const std::unordered_map<std::string, std::string>& options_map) override {
    dedup_subgraph = true;
  }

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN BN + ReLU optimization pass";
    auto property                  = std::make_shared<SgDNNLBNReLUProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_BN_RELU_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();

    std::ostringstream node_name;
    node_name << "sg_onednn_batch_norm_relu_" << std::to_string(subgraph_id);

    // Copy params from BatchNorm node into subgraph BatchNormReLU node
    BatchNormParam param;
    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->op() && node->op()->name == "BatchNorm") {
        param = nnvm::get<BatchNormParam>(node->attrs.parsed);
      }
    });

    n->attrs.name = node_name.str();
    n->attrs.op   = Op::Get("_sg_onednn_batch_norm");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(sym));
    param.SetAttrDict(&(n->attrs.dict));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgDNNLBNReLUSelector>(disable_bn_relu_);
    return selector;
  }

 private:
  bool disable_bn_relu_;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_BN_RELU_PROPERTY_H_
