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
 * \file dnnl_remove_casts_property.h
 * \brief Graph property for removing two unnecessary Cast operations
 *
 * ... -> Cast(bool) -> expand_dims -> Cast(bool) -> Cast(bool) -> ...
 *                                  ||
 *                                  \/
 *                ... -> Cast(bool) -> expand_dims -> ...
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_REMOVE_CASTS_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_REMOVE_CASTS_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <map>
#include <string>
#include <vector>

#include "operator/subgraph/common.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgDNNLRemoveCastsSelector : public SubgraphSelectorV2 {
 private:
  enum CastStatus { kStart, kExpand, kCast, kSuccess, kFail };
  CastStatus status_  = kStart;
  const char* boolStr = "bool";

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    if (seed_node.node->op() == Op::Get("expand_dims")) {
      status_ = kExpand;
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    if (input_node.node->op() != Op::Get("Cast")) {
      status_ = kFail;
    } else if (input_node.node->attrs.dict["dtype"] != boolStr) {
      status_ = kFail;
    }
    return false;
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    if (status_ == kFail || status_ == kSuccess || output_node.node->is_variable()) {
      return false;
    }
    if (output_node.node->op() == Op::Get("Cast")) {
      if (output_node.node->attrs.dict["dtype"] == boolStr) {
        status_ = status_ == kExpand ? kCast : kSuccess;
        return true;
      }
    }
    status_ = kFail;
    return false;
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    return status_ == kSuccess ? candidates : std::vector<BiDirectedNode*>(0);
  }
};

class SgDNNLRemoveCastsProperty : public SubgraphProperty {
 public:
  SgDNNLRemoveCastsProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "Remove casts optimization pass";
    auto property                  = std::make_shared<SgDNNLRemoveCastsProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_REMOVE_CASTS_PROPERTY", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs.op       = Op::Get("expand_dims");
    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->attrs.op == Op::Get("expand_dims")) {
        n->attrs.name         = node->attrs.name;
        n->attrs.dict["axis"] = node->attrs.dict["axis"];
      }
      return;
    });

    if (n->op()->attr_parser) {
      n->op()->attr_parser(&(n->attrs));
    }
    return n;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLRemoveCastsSelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_REMOVE_CASTS_PROPERTY_H_