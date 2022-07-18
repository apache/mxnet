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

/*!
  \file dnnl_fc_sum_fuse_property.h
  \brief For fusing FullyConnected operator with element-wise add.

  Element-wise add operator is replaced by DNNL FC "sum" post operator.
  It adds FC results to existing values in output. For quantized integer version
  this output is scaled to the proper range.
*/

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_FC_SUM_FUSE_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_FC_SUM_FUSE_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../tensor/matrix_op-inl.h"
#include "../common.h"
#include "dnnl_fc-inl.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

inline bool EndsWith(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) {
    return false;
  } else {
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }
}

class SgDNNLFCSumFuseSelector : public SubgraphSelectorV2 {
 private:
  bool quantized_;
  bool patternFound = false;

 public:
  explicit SgDNNLFCSumFuseSelector(bool quantized) : quantized_(quantized) {}

  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    const auto n = seed_node.node;
    if (n->op() == Op::Get("_sg_onednn_fully_connected") && seed_node.outputs.size() == 1) {
      auto const& fc_param = nnvm::get<DNNLFCFullParam>(n->attrs.parsed);
      if (!quantized_ || (fc_param.dnnl_param.quantized && !fc_param.dnnl_param.with_eltwise)) {
        // Start subgraph when fusing for floats (quantized_ is false for ONEDNN backend) or
        // when FC is already quantized (second pass for ONEDNN_QUANTIZE) but not already fused
        // with elemwise operator.
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
    if (patternFound || output_n->is_variable()) {
      return false;
    }

    // Find _contrib_quantized_elemwise_add or elemwise_add
    if (EndsWith(output_n->op()->name, "elemwise_add")) {
      if (quantized_) {
        auto const& fc_param = nnvm::get<DNNLFCFullParam>(cur_n->attrs.parsed);
        if (!fc_param.dnnl_param.enabled_float_output.has_value()) {
          // For quantized graph, when FC floating point output is not enabled elementwise add must
          // also be quantized (min and max value have to be already stored in elementwise add).
          CHECK_EQ(output_n->attrs.dict.count("min_calib_range"), 1);
        }
      }
      patternFound = true;
      return true;
    } else {
      return false;
    }
  }

  std::vector<BiDirectedNode*> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    if (patternFound) {
      return candidates;
    } else {
      return std::vector<BiDirectedNode*>(0);
    }
  }

  void Reset() override {
    patternFound = false;
  }
};

class SgDNNLFCSumFuseProperty : public SubgraphProperty {
 public:
  SgDNNLFCSumFuseProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN fuse FullyConnected with sum";
    auto property                  = std::make_shared<SgDNNLFCSumFuseProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FC_SUM", 0)) {
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
      if (sub_name == "_sg_onednn_fully_connected") {
        fc_node = node;
      } else if (EndsWith(sub_name, "elemwise_add")) {
        ew_add_node = node;
      }
    });

    CHECK_NOTNULL(fc_node);
    if (ew_add_node != nullptr) {
      CHECK_NOTNULL(fc_node->attrs.subgraphs[0]);
      auto subgraph_output_node = fc_node->attrs.subgraphs[0]->outputs[0].node;
      nnvm::Symbol new_sym;
      // Create a new elemwise_add node to not alter the original one.
      // It is needed in subgraph to properly calculate InferShape.
      nnvm::ObjectPtr n = nnvm::Node::Create();
      n->attrs.op       = Op::Get("elemwise_add");
      n->attrs.name     = ew_add_node->attrs.name;

      if (ew_add_node->inputs[0].node == fc_node) {
        n->inputs.emplace_back(subgraph_output_node);
        n->inputs.emplace_back(ew_add_node->inputs[1]);
      } else {
        n->inputs.emplace_back(ew_add_node->inputs[0]);
        n->inputs.emplace_back(subgraph_output_node);
      }
      new_sym.outputs.emplace_back(n);
      fc_node->attrs.subgraphs.clear();
      fc_node->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
      fc_node->attrs.dict["with_sum"] = "True";
      fc_node->op()->attr_parser(&(fc_node->attrs));
    }
    return fc_node;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    bool quantized = HasAttr("quantize") ? GetAttr<bool>("quantize") : false;
    auto selector  = std::make_shared<SgDNNLFCSumFuseSelector>(quantized);
    return selector;
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr subgraph_node,
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr     = nnvm::NodeEntry{subgraph_node, entry_ptr->index, 0};
    }
  }

  void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    auto sym             = subgraph_node->attrs.subgraphs[0];
    auto const& fc_param = nnvm::get<DNNLFCFullParam>(subgraph_node->attrs.parsed);
    std::unordered_set<const nnvm::Node*> node_set;
    DFSVisit(sym->outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable()) {
        return;
      }
      node_set.insert(node.get());
      if (EndsWith(node->op()->name, "elemwise_add")) {
        const size_t base_inputs = fc_param.default_param.no_bias ? 3 : 4;
        // Make sure fc output is the left operand of the add operator, if not:
        // - swap inputs of add operator
        // - switch add operands sequence to ensure that
        // the tensor (sum_tensor) to which FC output is added is the last input.
        if (node_set.count(node->inputs[1].node.get())) {
          // Example of input_entries reordering for channel-wise quantized graph:
          // sum_tensor.data    -->   fc.data
          // fc.data            -->   fc.weight0
          // fc.weight0         -->   fc.bias0
          // fc.bias0           -->   sum_tensor.data
          // fc_out.min         -->   fc_out.min
          // fc_out.max         -->   fc_out.max
          // sum_tensor.min     -->   sum_tensor.min
          // sum_tensor.max     -->   sum_tensor.max
          std::swap(node->inputs[0], node->inputs[1]);
          std::rotate(input_entries->begin(),
                      input_entries->begin() + 1,
                      input_entries->begin() + base_inputs);
          std::rotate(orig_input_entries->begin(),
                      orig_input_entries->begin() + 1,
                      orig_input_entries->begin() + base_inputs);
        } else {
          // Example of input_entries reordering for channel-wise quantized graph:
          // fc.data            -->   fc.data
          // fc.weight0         -->   fc.weight0
          // fc.bias0           -->   fc.bias0
          // fc_out.min         -->   sum_tensor.data
          // fc_out.max         -->   fc_out.min
          // sum_tensor.data    -->   fc_out.max
          // sum_tensor.min     -->   sum_tensor.min
          // sum_tensor.max     -->   sum_tensor.max
          const int not_rotated_end = (fc_param.dnnl_param.quantized &&
                                       !fc_param.dnnl_param.enabled_float_output.has_value()) ?
                                          2 :
                                          0;

          std::rotate(input_entries->begin() + base_inputs - 1,
                      input_entries->end() - 1 - not_rotated_end,
                      input_entries->end() - not_rotated_end);
          std::rotate(orig_input_entries->begin() + base_inputs - 1,
                      orig_input_entries->end() - 1 - not_rotated_end,
                      orig_input_entries->end() - not_rotated_end);
        }
      }
    });
    subgraph_node->inputs = *orig_input_entries;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_FC_SUM_FUSE_PROPERTY_H_
