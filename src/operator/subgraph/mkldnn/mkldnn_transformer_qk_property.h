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


#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_QK_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_QK_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <map>
#include <string>
#include <vector>
#include "../common.h"
#include "../../numpy/np_matrix_op-inl.h"
#include "../../contrib/transformer-inl.h"
#include "../../tensor/matrix_op-inl.h"
#include "mkldnn_common.h"
#include "mkldnn_subgraph_base-inl.h"
#include "mkldnn_transformer-inl.h"

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

class SgMKLDNNTransformerQKSelector : public SubgraphSelector {
  enum SelectStatus {
    kFail = 0,
    kStart,
    kFirstSwapAx,
    kSecondSwapAx,
    kFirstReshape,
    kSecondReshape,
    kSuccess
  };

/*
  kStart ---> kFirstSwapAx ---> kSecondSwapAx ---> kFirstReshape ---> kSecondReshape ---> kSuccess
  Each status except kStart is connected with kFail
*/

 private:
  SelectStatus status_;
  std::vector<const nnvm::Node *> matched_list_;

 public:
  bool Select(const nnvm::Node &n, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (n.op() == Op::Get("batch_dot")) {
      status_ = kStart;
      matched_list_.clear();
      matched_list_.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (status_ == kFail || status_ == kSuccess || new_node.is_variable())
      return false;

    switch (status_) {
      case kStart:
        if (new_node.op() == Op::Get("SwapAxis")) {
          if (CheckSwapAxisConditions(new_node)) {
            status_ = kFirstSwapAx;
            matched_list_.push_back(&new_node);
          }
          return true;
        }
      case kFirstSwapAx:
        if (new_node.op() == Op::Get("SwapAxis")) {
          if (CheckSwapAxisConditions(new_node)) {
            status_ = kSecondSwapAx;
            matched_list_.push_back(&new_node);
            return true;
          }
        }
      case kSecondSwapAx:
        if (new_node.op() == Op::Get("_npx_reshape")) {
          // input to reshape must be first or second output from split
          if (CheckReshapeConditions(new_node, 0) || CheckReshapeConditions(new_node, 1)) {
            status_ = kFirstReshape;
            matched_list_.push_back(&new_node);
            return true;
          }
        }
      case kFirstReshape:
        if (new_node.op() == Op::Get("_npx_reshape")) {
          if (CheckReshapeConditions(new_node, 0) || CheckReshapeConditions(new_node, 1)) {
            status_ = kSecondReshape;
            matched_list_.push_back(&new_node);
            return true;
          }
        }
      case kSecondReshape:
        if (new_node.op() == Op::Get("_split_v2")) {
          status_ = kSuccess;
          return true;
        }
      default:
        status_ = kFail;
        return false;
    }
      return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status_ == kFail) {
      return std::vector<nnvm::Node *>(0);
    } else {
      std::vector<nnvm::Node *> ret;
      for (auto i : matched_list_) {
        auto non_const_i = const_cast<nnvm::Node *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgMKLDNNTransformerQKSelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgMKLDNNTransformerQKProperty : public SubgraphProperty {
 public:
  SgMKLDNNTransformerQKProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN Transformer optimization pass";
    auto property = std::make_shared<SgMKLDNNTransformerQKProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_TRANSFORMER_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    std::string op_name;

    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if ((node->op() == Op::Get("_npx_reshape"))) {
        auto const &reshape_param =
            nnvm::get<NumpyXReshapeParam>(node->attrs.parsed);
        // set heads attribute - all necessary conditions are checked before
        n->attrs.dict["heads"] = std::to_string(reshape_param.newshape[2]);
      }
    });

    node_name << "_sg_mkldnn_selfatt_qk_" << subgraph_id;

    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_selfatt_qk");
    CHECK(n->attrs.op);
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNTransformerQKSelector>();
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::ObjectPtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }


  void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries)
                             const override {
    subgraph_node->inputs.resize(1);
    // split is not part of subgraph, skip split as input and
    // connect subgraph input with split input
    subgraph_node->inputs[0] = orig_input_entries->at(0).node->inputs[0];
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_QK_PROPERTY_H_
