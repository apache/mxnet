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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_BATCH_DOT_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_BATCH_DOT_PROPERTY_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>

#include "operator/tensor/dot-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgDNNLBatchDotSelector : public SubgraphSelector {
 public:
  bool Select(const nnvm::Node& n) override {
    return n.op() && n.op()->name == "batch_dot";
  }

  bool SelectInput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node& n, const nnvm::Node& new_node) override {
    return false;
  }
};

class SgDNNLBatchDotProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    static const std::string& name = "oneDNN Batch Dot optimization pass";
    auto property                  = std::make_shared<SgDNNLBatchDotProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_BATCH_DOT_FUSE", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym,
                                     const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();

    std::ostringstream node_name;
    node_name << "sg_dnnl_batch_dot_" << std::to_string(subgraph_id);

    DotParam param;
    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& node) {
      if (node->op() && node->op()->name == "batch_dot") {
        param = nnvm::get<DotParam>(node->attrs.parsed);
      }
    });

    n->attrs.name = node_name.str();
    n->attrs.op   = Op::Get("_sg_onednn_batch_dot");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(sym));
    n->attrs.dict["transpose_a"] = std::to_string(param.transpose_a);
    n->attrs.dict["transpose_b"] = std::to_string(param.transpose_b);
    n->attrs.dict["quantized"]   = "False";
    n->op()->attr_parser(&(n->attrs));

    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgDNNLBatchDotSelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_BATCH_DOT_PROPERTY_H_
