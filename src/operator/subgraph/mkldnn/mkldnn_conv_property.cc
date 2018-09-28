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
#include "../../nn/activation-inl.h"

namespace mxnet {
namespace op {
class SgMKLDNNConvSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kBN,
    kSum,
    kSuccess,
  };

 private:
  bool disable_all;
  bool disable_conv_bn;
  bool disable_conv_relu;
  bool disable_conv_sum;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;

 public:
  SgMKLDNNConvSelector(int dis_all, int dis_conv_bn, int dis_conv_relu, int dis_conv_sum)
      : disable_all(dis_all),
        disable_conv_bn(dis_conv_bn),
        disable_conv_relu(dis_conv_relu),
        disable_conv_sum(dis_conv_sum) {}

  bool Select(const nnvm::Node &n) override {
    if (n.op() && n.op()->name == "Convolution") {
      status = disable_all ? kSuccess : kStart;
      matched_list.clear();
      matched_list.push_back(&n);
      return true;
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
      while (matched_list.back() != &n) {
        matched_list.pop_back();
      }
      status = kSuccess;
      return false;
    }
    // Use status machine to do selection. The status change is
    // kStart -> kBN -> kSum -> kSuccess
    switch (status) {
      case kStart:
        if ((!disable_conv_bn) && new_node.op()->name == "BatchNorm") {
          matched_list.push_back(&new_node);
          status = kBN;
          return true;
        }
      case kBN:
        if ((!disable_conv_sum) && new_node.op()->name == "elemwise_add") {
          matched_list.push_back(&new_node);
          status = kSum;
          return true;
        }
      case kSum:
      default:
        if ((!disable_conv_relu) && new_node.op()->name == "Activation") {
          const ActivationParam &param =
              nnvm::get<ActivationParam>(new_node.attrs.parsed);
          if (param.act_type == activation::kReLU) {
            matched_list.push_back(&new_node);
            // If we find conv+relu, then we can't match bn anymore.
            if (status == kStart) status = kBN;
            return true;
          } else {
            status = kSuccess;
            return false;
          }
        }
        status = kSuccess;
        return false;
    }
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status == kFail) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }
};

class SgMKLDNNConvProperty : public SubgraphProperty {
 public:
  SgMKLDNNConvProperty() {
    disable_all = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_OPT", 0);
    disable_conv_bn = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FUSE_CONV_BN", 0);
    disable_conv_relu = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FUSE_CONV_RELU", 0);
    disable_conv_sum = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FUSE_CONV_SUM", 0);

    disable_all =
        disable_all && disable_conv_bn && disable_conv_relu && disable_conv_sum;
    if (disable_all) {
      LOG(INFO) << "MKLDNN Convolution optimization pass is disabled.";
    } else {
      LOG(INFO) << "Start to execute MKLDNN Convolution optimization pass.";
    }
  }
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgMKLDNNConvProperty>();
  }
  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(nnvm::NodeEntry{last_node, 0, 0});
    std::ostringstream node_name;
    node_name << "sg_mkldnn_";
    bool _with_sum = false;
    DFSVisit(new_sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &sub_name = node->op()->name;
      if (sub_name == "Convolution") {
        node_name << "conv_";
      } else if (sub_name == "BatchNorm") {
        node_name << "bn_";
        n->attrs.dict["with_bn"] = "true";
      } else if (sub_name == "elemwise_add") {
        node_name << "add_";
        n->attrs.dict["with_sum"] = "true";
        _with_sum = true;

      } else if (sub_name == "Activation") {
        node_name << "relu_";
        if (!_with_sum) {
          n->attrs.dict["with_relu"] = "true";
        } else {
          n->attrs.dict["with_postsum_relu"] = "true";
        }
      }
    });
    node_name << std::to_string(subgraph_id);
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_conv");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNConvSelector>(
        disable_all, disable_conv_bn, disable_conv_relu, disable_conv_sum);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      *output_entries->at(i) = nnvm::NodeEntry{n, 0, 0};
    }
  }

  void ConnectSubgraphInputs(
      const nnvm::NodePtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
    auto sym = n->attrs.subgraphs[0];
    std::unordered_set<const nnvm::Node *> node_sets;
    DFSVisit(sym->outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      node_sets.insert(node.get());
      if (node->op()->name == "elemwise_add") {
        // Make sure n is the left operand of sum, if not,
        // switch sum operands sequence to ensure that
        // the extra sum operand stays in the last of inputs.
        if (node_sets.count(node->inputs[1].node.get())) {
          auto tmp = node->inputs[1];
          node->inputs[1] = node->inputs[0];
          node->inputs[0] = tmp;
          std::rotate(input_entries->begin(), input_entries->begin() + 1,
                      input_entries->end());
          std::rotate(orig_input_entries->begin(),
                      orig_input_entries->begin() + 1,
                      orig_input_entries->end());
        }
      }
    });
    n->inputs = *orig_input_entries;
  }

 private:
  int disable_all;
  int disable_conv_bn;
  int disable_conv_relu;
  int disable_conv_sum;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN, SgMKLDNNConvProperty);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
