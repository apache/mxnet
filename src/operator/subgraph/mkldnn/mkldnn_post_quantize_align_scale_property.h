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
#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_POST_QUANTIZE_ALIGN_SCALE_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_POST_QUANTIZE_ALIGN_SCALE_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "../subgraph_property.h"

namespace mxnet {
namespace op {

class SgMKLDNNConcatPostQuantizeSelector : public SubgraphSelectorV2 {
 public:
  explicit SgMKLDNNConcatPostQuantizeSelector(int dis_all) : disable_all_(dis_all) {}

  bool Select(const SimpleNode &sn) override {
    const auto &n = *sn.node;
    if ((!disable_all_) && n.op() == Op::Get("_contrib_quantized_concat")) {
      matched_list_.clear();
      visit_list_.clear();
      visit_list_.insert(&n);
      select_output_ = (sn.outputs.size() > 1) ? false : true;
      return true;
    }
    return false;
  }

  bool SelectInput(const SimpleNode &sn, const SimpleNode &snew_node) override {
    const auto &n = *sn.node;
    const auto &new_node = *snew_node.node;
    if (new_node.is_variable()) return false;
    if (visit_list_.count(&n) == 0) return false;
    bool multiple_outputs = false;
    for (auto i : snew_node.outputs) {
      if (visit_list_.count(i.first) == 0) {
        multiple_outputs = true;
        break;
      }
    }
    if (multiple_outputs) return false;
    if (new_node.attrs.dict.count("min_calib_range") != 0 &&
        new_node.attrs.dict.count("max_calib_range") != 0) {
      matched_list_.push_back(&snew_node);
      return true;
    } else if (new_node.op() == Op::Get("_contrib_quantized_concat") ||
               new_node.op() == Op::Get("_contrib_quantized_pooling")) {
      visit_list_.insert(&new_node);
      return true;
    }
    return false;
  }

  bool SelectOutput(const SimpleNode &sn, const SimpleNode &snew_node) override {
    if (!select_output_) return false;
    const auto &n = *sn.node;
    const auto &new_node = *snew_node.node;
    if (new_node.is_variable()) return false;
    if (visit_list_.count(&n) == 0) {
      return false;
    }
    if (new_node.op() == Op::Get("_contrib_quantized_concat") ||
        new_node.op() == Op::Get("_contrib_quantized_pooling")) {
      visit_list_.insert(&new_node);
      return true;
    }
    return false;
  }

  virtual std::vector<SimpleNode*> Filter(const std::vector<SimpleNode*>& candidates) {
    if (matched_list_.size() < 2) {
      return std::vector<SimpleNode*>(0);
    } else {
      std::vector<SimpleNode *> ret;
      for (auto i : matched_list_) {
        ret.push_back(const_cast<SimpleNode *>(i));
      }
      return ret;
    }
  }

 private:
  bool disable_all_;
  bool select_output_;
  std::vector<const SimpleNode *> matched_list_;
  std::unordered_set<const nnvm::Node*> visit_list_;
};

class SgMKLDNNPostQuantizeAlignScaleProperty : public SubgraphProperty {
 public:
  SgMKLDNNPostQuantizeAlignScaleProperty() : SubgraphProperty(kAdjust) {
    disable_all_ = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_OPT", 0);
    if (disable_all_) {
      LOG(INFO) << "MKLDNN post-quantization scale alignment optimization pass is disabled.";
    } else {
      LOG(INFO) << "Start to execute MKLDNN post-quantization scale alignment optimization pass.";
    }
  }
  static SubgraphPropertyPtr Create() {
    auto property = std::make_shared<SgMKLDNNPostQuantizeAlignScaleProperty>();
    property->SetAttr<std::string>("prop_name",
                                   "MKLDNN post-quantization scale alignment optimization pass");
    property->SetAttr<bool>("inference_only", true);
    return property;
  }
  void AdjustSubgraphNode(const std::vector<nnvm::Node *> &subgraph_nodes,
                          const SubgraphSelectorV2Ptr &subgraph_selector,
                          const int subgraph_id = 0) const override {
    float min_calib = 0.0f;
    float max_calib = 0.0f;
    for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
      auto this_min_calib = std::stof(subgraph_nodes[i]->attrs.dict["min_calib_range"]);
      auto this_max_calib = std::stof(subgraph_nodes[i]->attrs.dict["max_calib_range"]);
      if (min_calib > this_min_calib) min_calib = this_min_calib;
      if (max_calib < this_max_calib) max_calib = this_max_calib;
    }
    for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
      auto &n = *subgraph_nodes[i];
      n.attrs.dict["min_calib_range"] = std::to_string(min_calib);
      n.attrs.dict["max_calib_range"] = std::to_string(max_calib);
      if (n.op()->attr_parser) n.op()->attr_parser(&(n.attrs));
    }
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgMKLDNNConcatPostQuantizeSelector>(disable_all_);
    return selector;
  }

 private:
  int disable_all_;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_POST_QUANTIZE_ALIGN_SCALE_PROPERTY_H_
