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
#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_AMP_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_AMP_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <set>
#include <string>
#include <vector>

#include "operator/tensor/amp_cast.h"
#include "operator/subgraph/common.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

inline bool IsSupportedAMPFuseOp(const nnvm::Node& node) {
  static const std::set<const Op*> supported_ops = {Op::Get("_sg_onednn_conv"),
                                                    Op::Get("_sg_onednn_fully_connected"),
                                                    Op::Get("_sg_onednn_selfatt_qk"),
                                                    Op::Get("_sg_onednn_selfatt_qk_split"),
                                                    Op::Get("_sg_onednn_selfatt_valatt")};
  return supported_ops.count(node.op()) > 0;
}

class SgDNNLPostAMPSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum class SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  SelectStatus status;
  std::vector<const nnvm::Node*> matched_list;

 public:
  bool Select(const nnvm::Node& n) override {
    if (IsSupportedAMPFuseOp(n)) {
      status = SelectStatus::kStart;
      matched_list.clear();
      matched_list.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    if (status == SelectStatus::kFail || new_node.is_variable())
      return false;
    // If 'n' is not the last matched node, then we have encountered an internal branch and we
    // should pop out the node behind 'n' and stop fusion.
    if (matched_list.back() != &n) {
      status = SelectStatus::kFail;
      return false;
    }
    if (status == SelectStatus::kSuccess) {
      return false;
    }
    if (new_node.op()->name == "amp_cast") {
      auto const& param = nnvm::get<AMPCastParam>(new_node.attrs.parsed);
      if (param.dtype == mshadow::kFloat32) {
        matched_list.push_back(&new_node);
        status = SelectStatus::kSuccess;
        return true;
      }
      status = SelectStatus::kFail;
    }
    return false;
  }

  std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) override {
    if (status != SelectStatus::kSuccess) {
      return std::vector<nnvm::Node*>(0);
    } else {
      return candidates;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list.size(), 1);
    auto new_selector = SgDNNLPostAMPSelector();
    new_selector.Select(*matched_list[0]);
    *this = new_selector;
  }
};

class SgDNNLPostAMPProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN post-amp optimization pass";
    auto property                  = std::make_shared<SgDNNLPostAMPProperty>();
    property->SetAttr<std::string>("property_name", name);
    return property;
  }
  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr fuse_node = nullptr;
    nnvm::ObjectPtr amp_node  = nullptr;
    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (IsSupportedAMPFuseOp(*node)) {
        fuse_node = node;
      } else if (node->op() == Op::Get("amp_cast")) {
        amp_node = node;
      }
    });
    CHECK_NOTNULL(fuse_node);
    CHECK_NOTNULL(amp_node);
    fuse_node->attrs.dict["enabled_float_output"] = amp_node->attrs.dict["dtype"];
    fuse_node->op()->attr_parser(&(fuse_node->attrs));
    return fuse_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgDNNLPostAMPSelector>();
    return selector;
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      output_entries->at(i)->node = n;
    }
  }
};
}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_AMP_PROPERTY_H_
