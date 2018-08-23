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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_CONV_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_CONV_H_

#if MXNET_USE_MKLDNN == 1

#include "../common.h"
#include "../subgraph_property.h"
#include "../../nn/convolution-inl.h"
#include "../../nn/activation-inl.h"

namespace mxnet {
namespace op {
namespace sg {

class SgMKLDNNConvSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    sFail = 0,
    sStart,
    sBN,
    sSum,
    sSuccess,
  };

 private:
  bool disable_conv_bn;
  bool disable_conv_relu;
  bool disable_conv_sum;
  bool disable_all;
  SelectStatus status;
  nnvm::NodeEntry conv_data;
  std::vector<const nnvm::Node *> matched_list;

  bool HandleMatchStatus() {
    if (matched_list.size() > 1) {
      status = sSuccess;
    } else {
      status = sFail;
    }
    return false;
  }

 public:
  SgMKLDNNConvSelector(int dis_conv_bn, int dis_conv_relu, int dis_conv_sum)
      : disable_conv_bn(dis_conv_bn),
        disable_conv_relu(dis_conv_relu),
        disable_conv_sum(dis_conv_sum),
        disable_all(disable_conv_bn && disable_conv_relu && disable_conv_sum) {}

  virtual bool Select(const nnvm::Node &n) override {
    bool match =
        (!disable_all) && (!n.is_variable()) && (n.op()->name == "Convolution");
    if (match) {
      status = sStart;
      conv_data = n.inputs[0];
      matched_list.clear();
      matched_list.push_back(&n);
      return true;
    }
    return false;
  }

  virtual bool SelectInput(const nnvm::Node &n,
                           const nnvm::Node &new_node) override {
    return false;
  }

  virtual bool SelectOutput(const nnvm::Node &n,
                            const nnvm::Node &new_node) override {
    if (status == sFail || status == sSuccess || new_node.is_variable())
      return false;
    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list.back() != &n) {
      while (matched_list.back() != &n) {
        matched_list.pop_back();
      }
      // If the remaining node is more than 1, then we can still do fusion.
      return HandleMatchStatus();
    }
    // Use status machine to do selection. The status change is
    // sStart -> sBN -> sSum -> sSuccess
    switch (status) {
      case sStart:
        if ((!disable_conv_bn) && new_node.op()->name == "BatchNorm") {
          matched_list.push_back(&new_node);
          status = sBN;
          return true;
        }
      case sBN:
        if ((!disable_conv_sum) && new_node.op()->name == "elemwise_add") {
          // Make sure n is the left operand of sum, if not,
          // switch sum operands sequence to ensure that
          // the extra sum operand stays in the last of inputs.
          auto sum_entry = new_node.inputs[1];
          if (new_node.inputs[1].node.get() == &n) {
            sum_entry = new_node.inputs[0];
          }
          #if 0
          if (sum_entry.node == conv_data.node &&
              sum_entry.index == conv_data.index) {
            // At this situation, we faced a structure like,
            // data -> conv -> sum
            //     \---------/
            // As conv+sum is a inplace operating, sum's output
            // will override data, which is not supported.
            return HandleMatchStatus();
          }
          #endif
          matched_list.push_back(&new_node);
          status = sSum;
          return true;
        }
      case sSum:
      default:
        if ((!disable_conv_relu) && new_node.op()->name == "Activation") {
          const ActivationParam &param =
              nnvm::get<ActivationParam>(new_node.attrs.parsed);
          if (param.act_type == activation::kReLU) {
            matched_list.push_back(&new_node);
            // If we find conv+relu, then we can't match bn anymore.
            if (status == sStart) status = sBN;
            return true;
          } else {
            return HandleMatchStatus();
          }
        }
        return HandleMatchStatus();
    }
  }

  virtual std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status == sFail || candidates.size() <= 1) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }
};

class SgMKLDNNConvProperty : public SubgraphProperty {
 public:
  SgMKLDNNConvProperty() {
    int disable_all = dmlc::GetEnv("MXNET_DISABLE_FUSION_ALL", 0);
    disable_conv_bn = dmlc::GetEnv("MXNET_DISABLE_FUSION_CONV_BN", 0);
    disable_conv_relu = dmlc::GetEnv("MXNET_DISABLE_FUSION_CONV_RELU", 0);
    disable_conv_sum = dmlc::GetEnv("MXNET_DISABLE_FUSION_CONV_SUM", 0);

    if (disable_all ||
        (disable_conv_bn && disable_conv_relu && disable_conv_sum)) {
      LOG(INFO) << "MKLDNN Convolution fusion pass is disabled. Fusion "
                   "configurations: ";
    } else {
      LOG(INFO) << "Start to execute MKLDNN Convolution fusion pass. Fusion "
                   "configurations:";
    }
    LOG(INFO) << "MXNET_DISABLE_FUSION_ALL=" << disable_all;
    LOG(INFO) << "MXNET_DISABLE_FUSION_CONV_BN=" << disable_conv_bn;
    LOG(INFO) << "MXNET_DISABLE_FUSION_CONV_RELU=" << disable_conv_relu;
    LOG(INFO) << "MXNET_DISABLE_FUSION_CONV_SUM=" << disable_conv_sum;
    if (disable_all) {
      disable_conv_bn = 1;
      disable_conv_relu = 1;
      disable_conv_sum = 1;
    }
  }
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgMKLDNNConvProperty>();
  }
  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    // Initialize new attributes to false
    n->attrs.dict["in_sum_at_begin"] = "false";
    n->attrs.dict["no_bias"] = "false";
    n->attrs.dict["with_bn"] = "false";
    n->attrs.dict["with_sum"] = "false";
    n->attrs.dict["with_relu"] = "false";
    n->attrs.dict["with_postsum_relu"] = "false";
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(nnvm::NodeEntry{last_node, 0, 0});
    std::string node_name = "";
    bool _with_sum = false;
    std::unordered_set<const nnvm::Node*> node_sets;
    DFSVisit(new_sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      node_sets.insert(node.get());
      auto &sub_name = node->op()->name;
      if (sub_name == "Convolution") {
        node_name += "Conv_";
        const ConvolutionParam &conv_params =
            nnvm::get<ConvolutionParam>(node->attrs.parsed);
        n->attrs.dict["no_bias"] = conv_params.no_bias ? "true" : "false";
      } else if (sub_name == "BatchNorm") {
        node_name += "BN_";
        n->attrs.dict["with_bn"] = "true";
      } else if (sub_name == "elemwise_add") {
        node_name += "Add_";
        n->attrs.dict["with_sum"] = "true";
        _with_sum = true;
        if (node_sets.count(node->inputs[1].node.get())) {
          n->attrs.dict["in_sum_at_begin"] = "true";
        } else {
          CHECK_NE(node_sets.count(node->inputs[0].node.get()), 0U);
        }
      } else if (sub_name == "Activation") {
        node_name += "Relu_";
        if (!_with_sum) {
          n->attrs.dict["with_relu"] = "true";
        } else {
          n->attrs.dict["with_postsum_relu"] = "true";
        }
      }
    });

    n->attrs.name = "sg_mkldnn_" + node_name + std::to_string(subgraph_id);
    n->attrs.op = Op::Get("_sg_mkldnn_conv");
    CHECK(n->attrs.op);
    n->attrs.parsed = new_sym;
    return n;
  }

  virtual SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNConvSelector>(
        disable_conv_bn, disable_conv_relu, disable_conv_sum);
    return selector;
  }

  virtual void ConnectSubgraphOutput(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> &output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries.size(); ++i) {
      *output_entries[i] = nnvm::NodeEntry{n, 0, 0};
    }
  }

 private:
  int disable_conv_bn;
  int disable_conv_relu;
  int disable_conv_sum;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN, SgMKLDNNConvProperty);

} // namespace sg
} // namespace op
} // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_CONV_H_
