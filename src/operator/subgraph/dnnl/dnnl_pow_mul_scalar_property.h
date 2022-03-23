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
 * \file dnnl_pow_mul_scalar_property.h
 * \brief Graph property for fusing _npi_power_scalar with _npi_multiply_scalar
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POW_MUL_SCALAR_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POW_MUL_SCALAR_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <map>
#include <string>
#include <vector>

#include "operator/subgraph/common.h"
#include "operator/tensor/elemwise_binary_scalar_op.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgDNNLPowMulScalarSelector : public SubgraphSelectorV2 {
 private:
  std::vector<const BiDirectedNode*> matched_list_;
  SelectStatus status_;

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    if (seed_node.node->op() == Op::Get("_npi_power_scalar")) {
      matched_list_.clear();
      matched_list_.emplace_back(&seed_node);
      status_ = kStart;
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    return false;
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    const nnvm::Node* raw_power_scalar_node = n.node;
    const nnvm::Node* raw_next_node         = output_node.node;
    if (raw_power_scalar_node->op() && raw_power_scalar_node->op()->name == "_npi_power_scalar") {
      if (raw_next_node->op() && status_ == kStart &&
          raw_next_node->op()->name == "_npi_multiply_scalar") {
        status_ = kSuccess;
        return true;
      } else {
        status_ = kFail;
        return false;
      }
    }

    if (matched_list_.back() != &n) {
      if (std::find(matched_list_.begin(), matched_list_.end(), &n) != matched_list_.end()) {
        while (matched_list_.back() != &n) {
          matched_list_.pop_back();
        }
      }
      status_ = kSuccess;
      return false;
    }

    return false;
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgDNNLPowMulScalarSelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgDNNLPowMulScalarProperty : public SubgraphProperty {
 public:
  SgDNNLPowMulScalarProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string& name = "DNNL PowMulScalar optimization pass";
    auto property                  = std::make_shared<SgDNNLPowMulScalarProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_POW_MUL_SCALAR_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();

    std::ostringstream node_name;
    node_name << "sg_dnnl_pow_mul_scalar_" << std::to_string(subgraph_id);

    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable())
        return;
      auto& sub_name = node->op()->name;
      if (sub_name == "_npi_power_scalar") {
        n->attrs.dict["exponent"] =
            std::to_string(nnvm::get<NumpyBinaryScalarParam>(node->attrs.parsed).scalar);
      } else if (sub_name == "_npi_multiply_scalar") {
        n->attrs.dict["multiplier"] =
            std::to_string(nnvm::get<NumpyBinaryScalarParam>(node->attrs.parsed).scalar);
      }
    });

    n->attrs.name = node_name.str();
    n->attrs.op   = Op::Get("_sg_onednn_pow_mul_scalar");
    CHECK(n->attrs.op);
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgDNNLPowMulScalarSelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_POW_MUL_SCALAR_PROPERTY_H_
