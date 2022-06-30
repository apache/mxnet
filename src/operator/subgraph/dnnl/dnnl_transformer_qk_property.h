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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <map>
#include <string>
#include <vector>

#include "operator/contrib/transformer-inl.h"
#include "operator/numpy/np_matrix_op-inl.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_common.h"
#include "dnnl_subgraph_base-inl.h"
#include "dnnl_transformer-inl.h"

/*
              custom_op
                 |
    _____________|_________________
   |            Split             |
   |           /     \            |
   |  _npx_reshape  _npx_reshape  |
   |      |              |        |
   |  SwapAxis        SwapAxis    |
   |       \          /           |
   |        batch_dot             |
   |            |                 |
   |______________________________|
*/
namespace mxnet {
namespace op {

class SgDNNLTransformerQKSelector : public SubgraphSelectorV2 {
  enum SelectStatusTransformerQK {
    kFail = 0,
    kStart,
    kFirstSwapAx,
    kSecondSwapAx,
    kFirstReshape,
    kSecondReshape,
    kSuccess
  };

  /*!
    kStart ---> kFirstSwapAx ---> kSecondSwapAx ---> kFirstReshape ---> kSecondReshape ---> kSuccess
    Each status except kStart is connected with kFail
  */

 private:
  SelectStatusTransformerQK status_;
  std::vector<const BiDirectedNode*> matched_list_;

  bool CheckSplitConditions(const BiDirectedNode& node) {
    const SplitParam& param = dmlc::get<SplitParam>(node.node->attrs.parsed);

    if (param.axis != -1 || param.sections != 3 || param.squeeze_axis)
      return false;

    const auto first_reshape  = (*(matched_list_.end() - 2))->node;
    const auto second_reshape = (*(matched_list_.end() - 1))->node;
    if (first_reshape->op() != Op::Get("_npx_reshape") ||
        second_reshape->op() != Op::Get("_npx_reshape")) {
      return false;
    }
    // 3 sections - ensure that every output is used only once
    if (node.outputs.size() == 3 && node.outputs.count(first_reshape) &&
        node.outputs.count(second_reshape)) {
      return true;
    }

    return false;
  }

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    if (seed_node.node->op() == Op::Get("batch_dot")) {
      status_ = kStart;
      matched_list_.clear();
      matched_list_.push_back(&seed_node);
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    if (status_ == kFail || status_ == kSuccess || input_node.node->is_variable())
      return false;
    const auto& raw_input_node = *input_node.node;
    switch (status_) {
      case kStart:
        if (raw_input_node.op() == Op::Get("SwapAxis")) {
          if (CheckSwapAxisConditions(raw_input_node)) {
            status_ = kFirstSwapAx;
            matched_list_.push_back(&input_node);
          }
          return true;
        }
      case kFirstSwapAx:
        if (raw_input_node.op() == Op::Get("SwapAxis")) {
          if (CheckSwapAxisConditions(raw_input_node)) {
            status_ = kSecondSwapAx;
            matched_list_.push_back(&input_node);
            return true;
          }
        }
      case kSecondSwapAx:
        if (raw_input_node.op() == Op::Get("_npx_reshape")) {
          // input to reshape must be first or second output from split
          if (CheckReshapeConditions(raw_input_node, 0) ||
              CheckReshapeConditions(raw_input_node, 1)) {
            status_ = kFirstReshape;
            matched_list_.push_back(&input_node);
            return true;
          }
        }
      case kFirstReshape:
        if (raw_input_node.op() == Op::Get("_npx_reshape")) {
          if (CheckReshapeConditions(raw_input_node, 0) ||
              CheckReshapeConditions(raw_input_node, 1)) {
            status_ = kSecondReshape;
            matched_list_.push_back(&input_node);
            return true;
          }
        }
      case kSecondReshape:
        if (raw_input_node.op() == Op::Get("_split_v2") && CheckSplitConditions(input_node)) {
          status_ = kSuccess;
          return true;
        }
      default:
        status_ = kFail;
        return false;
    }
    return false;
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    return false;
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    if (status_ != kSuccess) {
      return std::vector<BiDirectedNode*>(0);
    } else {
      std::vector<BiDirectedNode*> ret;
      for (auto i : matched_list_) {
        auto non_const_i = const_cast<BiDirectedNode*>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) != candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgDNNLTransformerQKSelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgDNNLTransformerQKProperty : public SubgraphProperty {
 public:
  SgDNNLTransformerQKProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN Transformer optimization pass";
    auto property                  = std::make_shared<SgDNNLTransformerQKProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_TRANSFORMER_OPT", 0)) {
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
    std::string op_name;

    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if ((node->op() == Op::Get("_npx_reshape"))) {
        auto const& reshape_param = nnvm::get<NumpyXReshapeParam>(node->attrs.parsed);
        // set heads attribute - all necessary conditions are checked before
        n->attrs.dict["heads"] = std::to_string(reshape_param.newshape[2]);
      }
    });

    node_name << "_sg_onednn_selfatt_qk_" << subgraph_id;

    n->attrs.name = node_name.str();
    n->attrs.op   = Op::Get("_sg_onednn_selfatt_qk");
    CHECK(n->attrs.op);
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLTransformerQKSelector>();
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

  void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    subgraph_node->inputs.resize(1);
    // split is not part of subgraph, skip split as input and
    // connect subgraph input with split input
    subgraph_node->inputs[0] = orig_input_entries->at(0).node->inputs[0];
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_PROPERTY_H_
