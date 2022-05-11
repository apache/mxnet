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
  SelectStatus status_ = kStart;

 public:
  bool Select(const BiDirectedNode& seed_node,
              const std::shared_ptr<NodeAttr>& node_attr) override {
    if (seed_node.node->op() == Op::Get("_npi_power_scalar") &&
        seed_node.node->num_outputs() == 1) {
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode& n, const BiDirectedNode& input_node) override {
    return false;
  }

  bool SelectOutput(const BiDirectedNode& n, const BiDirectedNode& output_node) override {
    if (output_node.node->op() == Op::Get("_npi_multiply_scalar") &&
        output_node.node->num_inputs() == 1 && status_ == kStart) {
      status_ = kSuccess;
      return true;
    }
    status_ = kFail;
    return false;
  }

  void Reset() override {
    status_ = kStart;
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

    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->is_variable())
        return;
      if (node->op() == Op::Get("_npi_power_scalar")) {
        n->attrs.dict["exponent"] =
            std::to_string(nnvm::get<NumpyBinaryScalarParam>(node->attrs.parsed).scalar);
      } else if (node->op() == Op::Get("_npi_multiply_scalar")) {
        n->attrs.dict["multiplier"] =
            std::to_string(nnvm::get<NumpyBinaryScalarParam>(node->attrs.parsed).scalar);
      }
    });

    n->attrs.name = "sg_dnnl_pow_mul_scalar_" + std::to_string(subgraph_id);
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
