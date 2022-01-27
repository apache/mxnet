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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_QUANTIZE_ALIGN_SCALE_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_QUANTIZE_ALIGN_SCALE_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>

#include "operator/subgraph/common.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgDNNLConcatPostQuantizeSelector : public SubgraphSelectorV2 {
 public:
  bool Select(const BiDirectedNode& sn) override {
    const auto& n = *sn.node;
    if (n.op() == Op::Get("_contrib_quantized_concat")) {
      head_ = sn;
      matched_list_.clear();
      visit_list_.clear();
      visit_list_.insert(&n);
      select_output_ = (sn.outputs.size() > 1) ? false : true;
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& sn, const BiDirectedNode& snew_node) override {
    const auto& n        = *sn.node;
    const auto& new_node = *snew_node.node;
    if (new_node.is_variable())
      return false;
    if (visit_list_.count(&n) == 0)
      return false;
    bool multiple_outputs = false;
    for (auto i : snew_node.outputs) {
      if (visit_list_.count(i.first) == 0) {
        multiple_outputs = true;
        break;
      }
    }
    if (multiple_outputs)
      return false;
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

  bool SelectOutput(const BiDirectedNode& sn, const BiDirectedNode& snew_node) override {
    if (!select_output_)
      return false;
    const auto& n        = *sn.node;
    const auto& new_node = *snew_node.node;
    if (new_node.is_variable())
      return false;
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

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    if (matched_list_.size() < 2) {
      return std::vector<BiDirectedNode*>(0);
    } else {
      std::vector<BiDirectedNode*> ret;
      for (auto i : matched_list_) {
        ret.push_back(const_cast<BiDirectedNode*>(i));
      }
      return ret;
    }
  }

  void Reset() override {
    auto new_selector = SgDNNLConcatPostQuantizeSelector();
    new_selector.Select(head_);
    *this = new_selector;
  }

 private:
  BiDirectedNode head_;
  bool select_output_;
  std::vector<const BiDirectedNode*> matched_list_;
  std::unordered_set<const nnvm::Node*> visit_list_;
};

class SgDNNLPostQuantizeAlignScaleProperty : public SubgraphProperty {
 public:
  SgDNNLPostQuantizeAlignScaleProperty() : SubgraphProperty(kAdjust) {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN post-quantization scale alignment optimization pass";
    auto property                  = std::make_shared<SgDNNLPostQuantizeAlignScaleProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }

  /*!
   * \brief Adjust selected nodes calibration range with maximum calib range.
   * For example,
   * conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
   *                               kernel=(3, 3), stride=(1, 1), no_bias=True)
   * conv2 = mx.symbol.Convolution(data=data, weight=weight * 2, name='conv2', num_filter=64,
   *                               kernel=(3, 3), stride=(1, 1), no_bias=True)
   * conv3 = mx.symbol.Convolution(data=data, weight=weight * 3, name='conv3', num_filter=64,
   *                               kernel=(3, 3), stride=(1, 1), no_bias=True)
   * conv4 = mx.symbol.Convolution(data=data, weight=weight * 4, name='conv4', num_filter=64,
   *                               kernel=(3, 3), stride=(1, 1), no_bias=True)
   * concat = mx.symbol.Concat(*[conv1, conv2, conv3, conv4], name="concat", dim=1)
   *
   * This pass will collect the maximum calib range from conv1 to conv4, and apply it to all
   * conv1 to conv4. Then concat don't need extra scale alignment operation. Performance and
   * accuracy are both improved.
   */
  void AdjustSubgraphNode(const std::vector<nnvm::Node*>& subgraph_nodes,
                          const SubgraphSelectorV2Ptr& subgraph_selector,
                          const int subgraph_id = 0) const override {
    float min_calib = 0.0f;
    float max_calib = 0.0f;
    for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
      auto this_min_calib = std::stof(subgraph_nodes[i]->attrs.dict["min_calib_range"]);
      auto this_max_calib = std::stof(subgraph_nodes[i]->attrs.dict["max_calib_range"]);
      if (min_calib > this_min_calib)
        min_calib = this_min_calib;
      if (max_calib < this_max_calib)
        max_calib = this_max_calib;
    }
    for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
      auto& n                         = *subgraph_nodes[i];
      n.attrs.dict["min_calib_range"] = std::to_string(min_calib);
      n.attrs.dict["max_calib_range"] = std::to_string(max_calib);
      if (n.op()->attr_parser)
        n.op()->attr_parser(&(n.attrs));
    }
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLConcatPostQuantizeSelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_QUANTIZE_ALIGN_SCALE_PROPERTY_H_
