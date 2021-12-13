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
#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_QUANTIZE_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_QUANTIZE_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "operator/nn/dnnl/dnnl_convolution-inl.h"
#include "operator/nn/fully_connected-inl.h"
#include "operator/quantization/requantize-inl.h"
#include "operator/tensor/elemwise_binary_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_conv-inl.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {
namespace {
const std::set<std::string> support_req_fusion_op = {"_contrib_quantized_elemwise_add",
                                                     "_contrib_quantized_elemwise_mul",
                                                     //"_contrib_quantized_npi_add",  // to be added later on
                                                     "_sg_onednn_conv",
                                                     "_sg_onednn_fully_connected",
                                                     "_sg_onednn_selfatt_qk",
                                                     "_sg_onednn_selfatt_valatt",
                                                     "_sg_onednn_batch_dot"};
}  // namespace

class SgDNNLPostQuantizeSelector : public SubgraphSelectorV2 {
 private:
  /*! \brief pattern match status */
  enum class SelectStatus {
    kFail = 0,
    kStart,
    kRequantize,
    kSuccess,
  };

  bool fuse_all;
  bool float_output;
  SelectStatus status;
  std::vector<const BiDirectedNode*> matched_list;
  std::set<std::string> support_requantize_fusion_op_name;

 public:
  explicit SgDNNLPostQuantizeSelector(const bool fuse_all, const bool float_output)
      : fuse_all(fuse_all), float_output(float_output) {
    support_requantize_fusion_op_name = support_req_fusion_op;
  }

  bool Select(const BiDirectedNode& n) override {
    const nnvm::Node* raw_node = n.node;
    if (fuse_all && raw_node->op() &&
        support_requantize_fusion_op_name.count(raw_node->op()->name)) {
      status = SelectStatus::kStart;
      matched_list.clear();
      matched_list.emplace_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& new_node) override {
    return false;
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& new_node) override {
    const nnvm::Node* raw_node     = n.node;
    const nnvm::Node* raw_new_node = new_node.node;
    if (status == SelectStatus::kFail || status == SelectStatus::kSuccess ||
        raw_new_node->is_variable())
      return false;
    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list.back() != &n) {
      if (std::find(matched_list.begin(), matched_list.end(), &n) != matched_list.end()) {
        while (matched_list.back() != &n) {
          matched_list.pop_back();
        }
      }
      status = SelectStatus::kSuccess;
      return false;
    }

    switch (status) {
      case SelectStatus::kStart:
        if (raw_new_node->op() == Op::Get("_contrib_requantize")) {
          auto const& param = nnvm::get<RequantizeParam>(raw_new_node->attrs.parsed);
          if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
            matched_list.emplace_back(&new_node);
            status = SelectStatus::kRequantize;
            if (no_enable_float_output.count(raw_node->op()) == 0) {
              status = SelectStatus::kSuccess;
            }
            return true;
          }
        }
      case SelectStatus::kRequantize:
        if (float_output && raw_new_node->op() == Op::Get("_contrib_dequantize")) {
          CHECK(raw_node->op() == Op::Get("_contrib_requantize"));
          if (n.outputs.size() > 1) {
            // check if requantize have other outputs than dequantize
            // if it has we can't fuse dequantize
            for (const auto& kv : n.outputs) {
              const auto& node = kv.first;
              if (node->op() != Op::Get("_contrib_dequantize")) {
                status = SelectStatus::kSuccess;
                return false;
              }
            }
          }
          matched_list.emplace_back(&new_node);
          status = SelectStatus::kSuccess;
          return true;
        }
      default:
        status = SelectStatus::kSuccess;
        return false;
    }
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    if (status != SelectStatus::kSuccess || (matched_list.size() <= 1)) {
      return std::vector<BiDirectedNode*>(0);
    } else {
      std::vector<BiDirectedNode*> ret;
      for (auto i : matched_list) {
        auto non_const_i = const_cast<BiDirectedNode*>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) != candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list.size(), 1);
    auto new_selector = SgDNNLPostQuantizeSelector(fuse_all, float_output);
    new_selector.Select(*matched_list[0]);
    *this = new_selector;
  }
};

class SgDNNLPostQuantizeProperty : public SubgraphProperty {
 private:
  bool fuse_all;
  bool float_output;
  std::set<std::string> support_requantize_fusion_op_name;

 public:
  SgDNNLPostQuantizeProperty() {
    fuse_all                          = dmlc::GetEnv("MXNET_ONEDNN_FUSE_REQUANTIZE", true);
    float_output                      = dmlc::GetEnv("MXNET_ONEDNN_FUSE_DEQUANTIZE", true);
    support_requantize_fusion_op_name = support_req_fusion_op;
  }

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN post-quantization optimization pass";
    auto property                  = std::make_shared<SgDNNLPostQuantizeProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr fuse_node       = nullptr;
    nnvm::ObjectPtr requantize_node = nullptr;
    nnvm::ObjectPtr dequantize_node = nullptr;
    const static std::set<const Op*> no_enable_float_output = {
      Op::Get("_contrib_quantized_elemwise_add")};

    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable())
        return;
      if (node->op() && support_requantize_fusion_op_name.count(node->op()->name)) {
        fuse_node = node;
      } else if (node->op() == Op::Get("_contrib_requantize")) {
        requantize_node = node;
      } else if (node->op() == Op::Get("_contrib_dequantize")) {
        dequantize_node = node;
      }
    });

    CHECK_NOTNULL(fuse_node);
    CHECK_NOTNULL(requantize_node);
    auto const& requantize_param = nnvm::get<RequantizeParam>(requantize_node->attrs.parsed);
    CHECK(requantize_param.min_calib_range.has_value());
    CHECK(requantize_param.max_calib_range.has_value());

    // When only fused quantized operator and requantize, set min/max_cablib_range,
    // When fused quantized operator + requantize + dequantize, set dequantize flag to true.
    if ((dequantize_node != nullptr) && (no_enable_float_output.count(fuse_node->op()) == 0)) {
      fuse_node->attrs.dict["enable_float_output"] = "True";
    } else {
      fuse_node->attrs.dict["min_calib_range"] =
          std::to_string(requantize_param.min_calib_range.value());
      fuse_node->attrs.dict["max_calib_range"] =
          std::to_string(requantize_param.max_calib_range.value());
    }
    fuse_node->op()->attr_parser(&(fuse_node->attrs));
    return fuse_node;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLPostQuantizeSelector>(fuse_all, float_output);
    return selector;
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr     = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POST_QUANTIZE_PROPERTY_H_
