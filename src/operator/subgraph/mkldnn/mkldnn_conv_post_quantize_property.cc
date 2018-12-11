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

#if MXNET_USE_MKLDNN == 1

#include "../common.h"
#include "../subgraph_property.h"
#include "../../nn/mkldnn/mkldnn_convolution-inl.h"
#include "mkldnn_conv-inl.h"
#include "../../quantization/requantize-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNConvPostQuantizeSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  bool disable_all;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;

 public:
  explicit SgMKLDNNConvPostQuantizeSelector(int dis_all)
      : disable_all(dis_all) {}

  bool Select(const nnvm::Node &n) override {
    if ((!disable_all) && n.op() && n.op()->name == "_sg_mkldnn_conv") {
      auto const &param = nnvm::get<MKLDNNConvFusionParam>(n.attrs.parsed);
      if (param.full_conv_param.mkldnn_param.quantized) {
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
};

class SgMKLDNNConvPostQuantizeProperty : public SubgraphProperty {
 public:
  SgMKLDNNConvPostQuantizeProperty() {
    disable_all = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_OPT", 0);
    if (disable_all) {
      LOG(INFO) << "MKLDNN Convolution post-quantization optimization pass is disabled.";
    } else {
      LOG(INFO) << "Start to execute MKLDNN Convolution post-quantization optimization pass.";
    }
  }
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgMKLDNNConvPostQuantizeProperty>();
  }
  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr conv_node = nullptr;
    nnvm::NodePtr requantize_node = nullptr;
    DFSVisit(sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &op_name = node->op()->name;
      if (op_name == "_sg_mkldnn_conv") {
        conv_node = node;
      } else if (op_name == "_contrib_requantize") {
        requantize_node = node;
      }
    });
    CHECK_NOTNULL(conv_node);
    CHECK_NOTNULL(requantize_node);
    auto const &requantize_param =
        nnvm::get<RequantizeParam>(requantize_node->attrs.parsed);
    CHECK(requantize_param.min_calib_range.has_value());
    CHECK(requantize_param.max_calib_range.has_value());
    conv_node->attrs.dict["min_calib_range"] =
        std::to_string(requantize_param.min_calib_range.value());
    conv_node->attrs.dict["max_calib_range"] =
        std::to_string(requantize_param.max_calib_range.value());
    conv_node->op()->attr_parser(&(conv_node->attrs));
    return conv_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector =
        std::make_shared<SgMKLDNNConvPostQuantizeSelector>(disable_all);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

 private:
  int disable_all;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_POST_QUANTIZE, SgMKLDNNConvPostQuantizeProperty);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
