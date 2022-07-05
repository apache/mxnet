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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_CONV_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_CONV_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>

#include "operator/leaky_relu-inl.h"
#include "operator/nn/activation-inl.h"
#include "operator/nn/convolution-inl.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {
class SgDNNLConvSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status_ */
  enum SelectStatusConv {
    kFail = 0,
    kStart,
    kBN,
    kSum,
    kSuccess,
  };

 private:
  bool disable_all_;
  bool disable_conv_bn_;
  bool disable_conv_act_;
  bool disable_conv_sum_;
  bool quantize_;
  SelectStatusConv status_;
  std::vector<const nnvm::Node*> matched_list_;

 public:
  SgDNNLConvSelector(int dis_all, int dis_conv_bn, int dis_conv_act, int dis_conv_sum, int quantize)
      : disable_all_(dis_all),
        disable_conv_bn_(dis_conv_bn),
        disable_conv_act_(dis_conv_act),
        disable_conv_sum_(dis_conv_sum),
        quantize_(quantize) {}

  bool Select(const nnvm::Node& n, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (n.op() && n.op()->name == "Convolution") {
      const auto& param = nnvm::get<ConvolutionParam>(n.attrs.parsed);
      if ((param.kernel.ndim() == 2 || param.kernel.ndim() == 3) && SupportDNNLAttr(node_attr)) {
        status_ = disable_all_ ? kSuccess : kStart;
        matched_list_.clear();
        matched_list_.push_back(&n);
        return true;
      }
    }
    return false;
  }

  bool SelectInput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list_.back() != &n) {
      if (std::find(matched_list_.begin(), matched_list_.end(), &n) != matched_list_.end()) {
        while (matched_list_.back() != &n) {
          matched_list_.pop_back();
        }
      }
      status_ = kSuccess;
      return false;
    }
    if (status_ == kFail || status_ == kSuccess || new_node.is_variable())
      return false;

    // Use status_ machine to do selection. The status_ change is
    // kStart -> kBN -> kSum -> kSuccess
    const auto node_name = new_node.op()->name;
    switch (status_) {
      case kStart:
        if ((!disable_conv_bn_) && node_name == "BatchNorm") {
          matched_list_.push_back(&new_node);
          status_ = kBN;
          return true;
        }
      case kBN:
        if ((!disable_conv_sum_) && (node_name == "elemwise_add" || node_name == "_npi_add")) {
          matched_list_.push_back(&new_node);
          status_ = kSum;
          return true;
        }
      case kSum:
      default:
        if ((!disable_conv_act_) && node_name == "Activation") {
          const ActivationParam& param = nnvm::get<ActivationParam>(new_node.attrs.parsed);
          if ((quantize_ && SupportDNNLQuantizedAct(param)) ||
              (!quantize_ && SupportDNNLAct(param))) {
            matched_list_.push_back(&new_node);
            // not support conv+relu+sum yet.
            status_ = kSuccess;
            return true;
          }
        } else if ((!disable_conv_act_) && node_name == "LeakyReLU") {
          const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(new_node.attrs.parsed);
          if (param.act_type == leakyrelu::kLeakyReLU || param.act_type == leakyrelu::kGELU_ERF ||
              param.act_type == leakyrelu::kGELU_TANH) {
            matched_list_.push_back(&new_node);
            // not support conv+relu+sum yet.
            status_ = kSuccess;
            return true;
          }
        } else if ((!disable_conv_act_) && node_name == "clip") {
          if (!(quantize_ && (status_ == kSum))) {
            // TODO(zhennan): doesn't support int8 conv+sum+relu6 at moment. To support this, we
            // need to fuse conv+sum first, and calibrate with it. Then fuse int8 relu6 into fused
            // conv.
            const ClipParam& param = nnvm::get<ClipParam>(new_node.attrs.parsed);
            if (param.a_min == 0.f) {
              matched_list_.push_back(&new_node);
              // not support conv+relu+sum yet.
              status_ = kSuccess;
              return true;
            }
          }
        }
        status_ = kSuccess;
        return false;
    }
  }

  std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) override {
    if (status_ == kFail) {
      return std::vector<nnvm::Node*>(0);
    } else {
      std::vector<nnvm::Node*> ret;
      for (auto i : matched_list_) {
        auto non_const_i = const_cast<nnvm::Node*>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) != candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgDNNLConvSelector(
        disable_all_, disable_conv_bn_, disable_conv_act_, disable_conv_sum_, quantize_);
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgDNNLConvProperty : public SubgraphProperty {
 public:
  SgDNNLConvProperty() {
    disable_conv_bn_  = dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FUSE_CONV_BN", 0);
    disable_conv_act_ = dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU", 0);
    disable_conv_sum_ = dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FUSE_CONV_SUM", 0);

    disable_all_ = disable_conv_bn_ && disable_conv_act_ && disable_conv_sum_;
  }
  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN convolution optimization pass";
    auto property                  = std::make_shared<SgDNNLConvProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_CONV_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }
  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    node_name << "sg_onednn_";
    bool _with_sum = false;
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable())
        return;
      auto& sub_name = node->op()->name;
      if (sub_name == "Convolution") {
        node_name << "conv_";
      } else if (sub_name == "BatchNorm") {
        node_name << "bn_";
        n->attrs.dict["with_bn"] = "true";
      } else if (sub_name == "elemwise_add" || sub_name == "_npi_add") {
        node_name << "add_";
        n->attrs.dict["with_sum"] = "true";
        _with_sum                 = true;
      } else if (sub_name == "Activation" || sub_name == "LeakyReLU" || sub_name == "clip") {
        node_name << "act_";
        if (!_with_sum) {
          n->attrs.dict["with_act"] = "true";
        } else {
          n->attrs.dict["with_postsum_act"] = "true";
        }
      }
    });
    node_name << std::to_string(subgraph_id);
    n->attrs.name = node_name.str();
    n->attrs.op   = Op::Get("_sg_onednn_conv");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    bool quantize = HasAttr("quantize") ? GetAttr<bool>("quantize") : false;
    auto selector = std::make_shared<SgDNNLConvSelector>(
        disable_all_, disable_conv_bn_, disable_conv_act_, disable_conv_sum_, quantize);
    return selector;
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      *output_entries->at(i) = nnvm::NodeEntry{n, 0, 0};
    }
  }

  void ConnectSubgraphInputs(const nnvm::ObjectPtr n,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    auto sym = n->attrs.subgraphs[0];
    std::unordered_set<const nnvm::Node*> node_sets;
    nnvm::Node* conv_input = nullptr;
    DFSVisit(sym->outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable())
        return;
      node_sets.insert(node.get());
      if (node->op()->name == "Convolution") {
        conv_input = node->inputs[0].node.get();
      } else if (node->op()->name == "elemwise_add" || node->op()->name == "_npi_add") {
        if (dedup_subgraph && (conv_input == node->inputs[1].node.get() ||
                               conv_input == node->inputs[0].node.get())) {
          n->attrs.dict["dedup_sum"] = "true";
          n->op()->attr_parser(&(n->attrs));
          return;
        }
        // Make sure n is the left operand of sum, if not,
        // switch sum operands sequence to ensure that
        // the extra sum operand stays in the last of inputs.
        if (node_sets.count(node->inputs[1].node.get())) {
          auto tmp        = node->inputs[1];
          node->inputs[1] = node->inputs[0];
          node->inputs[0] = tmp;
          std::rotate(input_entries->begin(), input_entries->begin() + 1, input_entries->end());
          std::rotate(orig_input_entries->begin(),
                      orig_input_entries->begin() + 1,
                      orig_input_entries->end());
        }
      }
    });
    n->inputs = *orig_input_entries;
  }

 private:
  int disable_all_;
  int disable_conv_bn_;
  int disable_conv_act_;
  int disable_conv_sum_;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_CONV_PROPERTY_H_
