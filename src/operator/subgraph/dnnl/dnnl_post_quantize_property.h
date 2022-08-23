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
bool SupportsRequantizeFusion(const Op* op) {
  static const std::set<const Op*> support_requantize_fusion_ops = {
      Op::Get("_contrib_quantized_elemwise_add"),
      Op::Get("_contrib_quantized_elemwise_mul"),
      Op::Get("_contrib_quantized_npi_add"),
      Op::Get("_sg_onednn_conv"),
      Op::Get("_sg_onednn_fully_connected"),
      Op::Get("_sg_onednn_selfatt_qk"),
      Op::Get("_sg_onednn_selfatt_qk_split"),
      Op::Get("_sg_onednn_selfatt_valatt"),
      Op::Get("_sg_onednn_batch_dot")};

  return support_requantize_fusion_ops.count(op) > 0;
}
}  // namespace

class SgDNNLPostQuantizeSelector : public SubgraphSelectorV2 {
 private:
  /*! \brief pattern match status */
  enum class SelectStatusPostQuantize {
    kFail = 0,
    kStart,
    kRequantize,
    kSuccess,
  };

  bool fuse_all;
  bool float_output;
  SelectStatusPostQuantize status;
  std::vector<const BiDirectedNode*> matched_list;

 public:
  explicit SgDNNLPostQuantizeSelector(const bool fuse_all, const bool float_output)
      : fuse_all(fuse_all), float_output(float_output) {}

  bool Select(const BiDirectedNode& n) override {
    const nnvm::Node* raw_node = n.node;

    if (fuse_all && raw_node->op() && SupportsRequantizeFusion(raw_node->op())) {
      status = SelectStatusPostQuantize::kStart;
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

    static const std::set<const Op*> dequantize_fusion_unsupported_ops = {
        Op::Get("_contrib_quantized_elemwise_add"), Op::Get("_contrib_quantized_npi_add")};

    if (status == SelectStatusPostQuantize::kFail || status == SelectStatusPostQuantize::kSuccess ||
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
      status = SelectStatusPostQuantize::kSuccess;
      return false;
    }

    switch (status) {
      case SelectStatusPostQuantize::kStart:
        if (raw_new_node->op() == Op::Get("_contrib_requantize")) {
          auto const& param = nnvm::get<RequantizeParam>(raw_new_node->attrs.parsed);
          if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
            matched_list.emplace_back(&new_node);
            status = SelectStatusPostQuantize::kRequantize;
            // For now there is no support for dequantize fusion for some operators
            // so then we finish after finding requantize node:
            if (dequantize_fusion_unsupported_ops.count(raw_node->op()) != 0) {
              status = SelectStatusPostQuantize::kSuccess;
            }
            return true;
          }
        }
      case SelectStatusPostQuantize::kRequantize:
        if (float_output && raw_new_node->op() == Op::Get("_contrib_dequantize")) {
          CHECK(raw_node->op() == Op::Get("_contrib_requantize"));
          if (n.outputs.size() > 1) {
            // check if requantize have other outputs than dequantize
            // if it has we can't fuse dequantize
            for (const auto& kv : n.outputs) {
              const auto& node = kv.first;
              if (node->op() != Op::Get("_contrib_dequantize")) {
                status = SelectStatusPostQuantize::kSuccess;
                return false;
              }
            }
          }
          matched_list.emplace_back(&new_node);
          status = SelectStatusPostQuantize::kSuccess;
          return true;
        }
      default:
        status = SelectStatusPostQuantize::kSuccess;
        return false;
    }
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    if (status != SelectStatusPostQuantize::kSuccess || (matched_list.size() <= 1)) {
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

 public:
  SgDNNLPostQuantizeProperty() {
    fuse_all     = dmlc::GetEnv("MXNET_ONEDNN_FUSE_REQUANTIZE", true);
    float_output = dmlc::GetEnv("MXNET_ONEDNN_FUSE_DEQUANTIZE", true);
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

    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable())
        return;
      if (node->op() && SupportsRequantizeFusion(node->op())) {
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
    if (dequantize_node != nullptr) {
      fuse_node->attrs.dict["enabled_float_output"] = type_string(mshadow::kFloat32);
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
