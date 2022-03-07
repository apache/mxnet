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

/*
  \file
  \brief For fusing FullyConnected operator with element-wise add.

  Element-wise add operator is replaced by MKLDNN FC "sum" post operator.
  It adds FC results to existing values in output. For quantized integer version
  this output is scaled to the proper range.
*/

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_SUM_FUSE_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_SUM_FUSE_H_
#if MXNET_USE_MKLDNN == 1

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../tensor/matrix_op-inl.h"
#include "../common.h"
#include "mkldnn_fc-inl.h"
#include "mkldnn_subgraph_base-inl.h"

namespace mxnet {
namespace op {

inline bool EndsWith(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) {
    return false;
  } else {
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }
}

class SgMKLDNNFCSumFuseSelector : public SubgraphSelectorV2 {
 private:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

  bool quantized_;
  SelectStatus status_;
  std::vector<const BiDirectedNode*> matched_list_;

 public:
  explicit SgMKLDNNFCSumFuseSelector(bool quantized) : quantized_(quantized) {}

  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    const auto n = seed_node.node;
    if ((n->op() == Op::Get("_sg_mkldnn_fully_connected")) && SupportMKLDNNAttr(node_attr) &&
        (seed_node.outputs.size() == 1)) {
      auto& fc_param = nnvm::get<MKLDNNFCFullParam>(n->attrs.parsed);
      if (quantized_) {
        if (fc_param.mkldnn_param.enable_fuse_add) {
          // Do not fuse during first pass with MKLDNN_QUANTIZE backend (quantized_ = true)
          // and mark to not fuse during quantization for run with MKLDNN backend during
          // quantization
          n->attrs.dict["enable_fuse_add"]      = "False";
          fc_param.mkldnn_param.enable_fuse_add = false;
          return false;
        } else {
          // On second pass MKLDNN_QUANTIZE backend fusing should happened, so
          // set to true (default value) and remove from dictionary
          n->attrs.dict.erase("enable_fuse_add");
          fc_param.mkldnn_param.enable_fuse_add = true;
        }
      }
      // Do not fuse for quantization if already fused with element-wise operation
      const bool fuse = fc_param.mkldnn_param.enable_fuse_add &&
                        (!quantized_ || !fc_param.mkldnn_param.with_eltwise);
      if (fuse) {
        status_ = kStart;
        matched_list_.clear();
        matched_list_.push_back(&seed_node);
        return true;
      }
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& cur_node, const BiDirectedNode& input_node) override {
    return false;
  }

  bool SelectOutput(const BiDirectedNode& cur_node, const BiDirectedNode& output_node) override {
    const auto cur_n    = cur_node.node;
    const auto output_n = output_node.node;
    if (status_ == kFail || status_ == kSuccess || output_n->is_variable()) {
      return false;
    }
    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list_.back() != &cur_node) {
      if (std::find(matched_list_.begin(), matched_list_.end(), &cur_node) != matched_list_.end()) {
        while (matched_list_.back() != &cur_node) {
          matched_list_.pop_back();
        }
      }
      status_ = kSuccess;
      return false;
    }

    switch (status_) {
      case kStart:
        // Find _contrib_quantized_elemwise_add or elemwise_add
        if (EndsWith(output_n->op()->name, "elemwise_add")) {
          if (quantized_) {
            auto const& fc_param = nnvm::get<MKLDNNFCFullParam>(cur_n->attrs.parsed);
            if (!fc_param.mkldnn_param.enable_float_output) {
              // For quantized graph, when FC floating point output is not enabled
              // elementwise add must also be quantized (min and max value have to be already stored
              // in elementwise add).
              CHECK_EQ(output_n->attrs.dict.count("min_calib_range"), 1);
            }
          }
          matched_list_.push_back(&output_node);
          status_ = kSuccess;
          return true;
        }
      default:
        status_ = kSuccess;
        return false;
    }
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    if (status_ == kFail) {
      return std::vector<BiDirectedNode*>(0);
    } else {
      return candidates;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgMKLDNNFCSumFuseSelector(quantized_);
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgMKLDNNFCSumFuseProperty : public SubgraphProperty {
 public:
  SgMKLDNNFCSumFuseProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "MKLDNN fuse FullyConnected with sum";
    auto property                  = std::make_shared<SgMKLDNNFCSumFuseProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FC_SUM", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr fc_node     = nullptr;
    nnvm::ObjectPtr ew_add_node = nullptr;

    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable()) {
        return;
      }
      auto& sub_name = node->op()->name;
      if (sub_name == "_sg_mkldnn_fully_connected") {
        fc_node = node;
      } else if (EndsWith(sub_name, "elemwise_add")) {
        ew_add_node = node;
      }
    });

    CHECK_NOTNULL(fc_node);
    if (ew_add_node != nullptr) {
      CHECK_NOTNULL(fc_node->attrs.subgraphs[0]);
      auto fc_orginal = fc_node->attrs.subgraphs[0]->outputs[0].node;

      if (fc_orginal->op() == Op::Get("FullyConnected")) {
        nnvm::Symbol new_sym;
        // Create a new elemwise_add node to not alter the original one.
        // It is needed in subgraph to properly calculate InferShape.
        nnvm::ObjectPtr n = nnvm::Node::Create();
        n->attrs.op       = Op::Get("elemwise_add");
        n->attrs.name     = ew_add_node->attrs.name;

        if (ew_add_node->inputs[0].node == fc_node) {
          n->inputs.emplace_back(fc_orginal);
          n->inputs.emplace_back(ew_add_node->inputs[1]);
        } else {
          n->inputs.emplace_back(ew_add_node->inputs[0]);
          n->inputs.emplace_back(fc_orginal);
        }
        new_sym.outputs.emplace_back(n);
        fc_node->attrs.subgraphs.clear();
        fc_node->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
        fc_node->attrs.dict["with_sum"] = "True";
        fc_node->op()->attr_parser(&(fc_node->attrs));
      }
    }
    return fc_node;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    bool quantized = HasAttr("quantize") ? GetAttr<bool>("quantize") : false;
    auto selector  = std::make_shared<SgMKLDNNFCSumFuseSelector>(quantized);
    return selector;
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr     = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

  void ConnectSubgraphInputs(const nnvm::ObjectPtr n,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    auto sym             = n->attrs.subgraphs[0];
    auto const& fc_param = nnvm::get<MKLDNNFCFullParam>(n->attrs.parsed);
    std::unordered_set<const nnvm::Node*> node_sets;
    DFSVisit(sym->outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable()) {
        return;
      }
      node_sets.insert(node.get());
      if (EndsWith(node->op()->name, "elemwise_add")) {
        const size_t base_inputs = fc_param.default_param.no_bias ? 3 : 4;

        // Make sure n is the left operand of sum, if not,
        // switch sum operands sequence to ensure that
        // the extra sum operand stays in the last of inputs.
        if (node_sets.count(node->inputs[1].node.get())) {
          std::swap(node->inputs[0], node->inputs[1]);
          std::rotate(input_entries->begin(),
                      input_entries->begin() + 1,
                      input_entries->begin() + base_inputs);
          std::rotate(orig_input_entries->begin(),
                      orig_input_entries->begin() + 1,
                      orig_input_entries->begin() + base_inputs);
        } else {
          const int not_rotated_end =
              (fc_param.mkldnn_param.quantized && !fc_param.mkldnn_param.enable_float_output) ? 2
                                                                                              : 0;

          std::rotate(input_entries->begin() + base_inputs - 1,
                      input_entries->end() - 1 - not_rotated_end,
                      input_entries->end() - not_rotated_end);
          std::rotate(orig_input_entries->begin() + base_inputs - 1,
                      orig_input_entries->end() - 1 - not_rotated_end,
                      orig_input_entries->end() - not_rotated_end);
        }
      }
    });
    n->inputs = *orig_input_entries;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_SUM_FUSE_H_
