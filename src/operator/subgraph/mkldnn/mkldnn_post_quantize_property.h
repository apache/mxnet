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
#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_POST_QUANTIZE_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_POST_QUANTIZE_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <set>
#include <string>
#include <vector>
#include "../../nn/mkldnn/mkldnn_convolution-inl.h"
#include "../../quantization/requantize-inl.h"
#include "../common.h"
#include "mkldnn_conv-inl.h"
#include "mkldnn_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNPostQuantizeSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;
  std::set<std::string> support_requantize_fusion_op_name;

 public:
  SgMKLDNNPostQuantizeSelector() {
    support_requantize_fusion_op_name.insert("_sg_mkldnn_conv");
    support_requantize_fusion_op_name.insert("_contrib_quantized_elemwise_add");
  }

  bool Select(const nnvm::Node &n) override {
    if (n.op() && support_requantize_fusion_op_name.count(n.op()->name)) {
      if (n.op()->name == "_sg_mkldnn_conv") {
        auto const &param = nnvm::get<MKLDNNConvFusionParam>(n.attrs.parsed);
        if (param.full_conv_param.mkldnn_param.quantized) {
          status = kStart;
          matched_list.clear();
          matched_list.push_back(&n);
          return true;
        }
      } else if (n.op()->name == "_contrib_quantized_elemwise_add") {
        status = kStart;
        matched_list.clear();
        matched_list.push_back(&n);
        return true;
      }
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
      status = kFail;
      return false;
    }
    if (new_node.op()->name == "_contrib_requantize") {
      auto const &param = nnvm::get<RequantizeParam>(new_node.attrs.parsed);
      if (param.min_calib_range.has_value() &&
          param.max_calib_range.has_value()) {
        matched_list.push_back(&new_node);
        status = kSuccess;
        return true;
      } else {
        status = kFail;
      }
    }
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status != kSuccess) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list.size(), 1);
    auto new_selector = SgMKLDNNPostQuantizeSelector();
    new_selector.Select(*matched_list[0]);
    *this = new_selector;
  }
};

class SgMKLDNNPostQuantizeProperty : public SubgraphProperty {
 public:
  SgMKLDNNPostQuantizeProperty() {
    support_requantize_fusion_op_name.insert("_sg_mkldnn_conv");
    support_requantize_fusion_op_name.insert("_contrib_quantized_elemwise_add");
  }
  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN post-quantization optimization pass";
    auto property = std::make_shared<SgMKLDNNPostQuantizeProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }
  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::ObjectPtr fuse_node = nullptr;
    nnvm::ObjectPtr requantize_node = nullptr;
    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if (node->is_variable()) return;
      auto &op_name = node->op()->name;
      if (support_requantize_fusion_op_name.count(op_name)) {
        fuse_node = node;
      } else if (op_name == "_contrib_requantize") {
        requantize_node = node;
      }
    });
    CHECK_NOTNULL(fuse_node);
    CHECK_NOTNULL(requantize_node);
    auto const &requantize_param =
        nnvm::get<RequantizeParam>(requantize_node->attrs.parsed);
    CHECK(requantize_param.min_calib_range.has_value());
    CHECK(requantize_param.max_calib_range.has_value());
    fuse_node->attrs.dict["min_calib_range"] =
        std::to_string(requantize_param.min_calib_range.value());
    fuse_node->attrs.dict["max_calib_range"] =
        std::to_string(requantize_param.max_calib_range.value());
    fuse_node->op()->attr_parser(&(fuse_node->attrs));
    return fuse_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNPostQuantizeSelector>();
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::ObjectPtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

 private:
  std::set<std::string> support_requantize_fusion_op_name;
};
}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_POST_QUANTIZE_PROPERTY_H_
