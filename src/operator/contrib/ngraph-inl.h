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
 * Copyright (c) 2018 Intel Corporation
 * \file ngraph.cc
 * \brief ngraph subgraph property for mxnet
*/

#ifndef MXNET_OPERATOR_CONTRIB_NGRAPH_INL_H_
#define MXNET_OPERATOR_CONTRIB_NGRAPH_INL_H_

#if MXNET_USE_NGRAPH
#include <ngraph_compiler.h>
#include <ngraph_imperative.h>
#include <ngraph_nnvm_ops.h>
#include <ngraph_utils.h>

#include <vector>

#include "../subgraph/common.h"
#include "../subgraph/subgraph_property.h"

namespace mxnet {
namespace op {

// when built with NGRAPH we use this subgraph by default
static int ngraph_backend = setenv("MXNET_SUBGRAPH_BACKEND", "ngraph", 0);

class SgNgraphSelector : public SubgraphSelector {
 public:
  // Public methods to implement the subgraph selector API
  explicit SgNgraphSelector(ngraph_bridge::Compiler *compiler)
      : compiler_(compiler), valid(compiler_->get_node_map().size() > 0) {}

  bool Select(const nnvm::Node &n) override { return is_node_selected(n); }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return is_node_selected(n, &new_node);
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return is_node_selected(n, &new_node);
  }
  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) {
    if (candidates.size() == 1 && candidates[0]->inputs.size() == 0) {
      return std::vector<nnvm::Node *>();
    } else {
      return candidates;
    }
  }

 private:
  ngraph_bridge::Compiler *compiler_;
  const bool valid;
  // get_node is a utility function to translate NNVM Nodes to
  // the IR nodes inside the ngraph_bridge::Compiler, this is
  // primarily utilized to help determine nGraph support
  ngraph_bridge::NodePtr get_node(const nnvm::Node *n) {
    if (n) {
      auto &entry_map = compiler_->get_ngraph().entry_map_;
      ngraph_bridge::MapEntry tmp{compiler_->get_node_map().at(n).get(), 0};
      if (entry_map.count(tmp)) {
        return entry_map[tmp];
      }
    }
    return nullptr;
  }
  // is_node_selected queries the ngraph_bridge::Compiler to determine if both
  // current and next NNVM nodes are supported by nGraph.
  // This allows us to meld nGraph's analysis with PartitionGraph.
  bool is_node_selected(const nnvm::Node &n, const nnvm::Node *next = nullptr) {
    bool selected = false;
    if (!valid) return selected;

    auto nn = get_node(&n);
    auto nnext = get_node(next);

    selected = nn && nn->in_ngraph_;
    if (next) {
      selected =
          selected && nnext->in_ngraph_ && nn->subgraph_ == nnext->subgraph_;
    }
    return selected;
  }
};

class SgNgraphProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    if (ngraph_backend != 0 && ngraph_bridge::ngraph_log_verbose_detail) {
      LOG(WARNING) << "NGRAPH_BRIDGE: failed to set MXNET_SUBGRAPH_BACKEND"
                   << std::endl;
    }
    return std::make_shared<SgNgraphProperty>();
  }

  bool NeedGraphAttrs() const override { return true; }
  // Create a subgraph node based on a symbol
  nnvm::NodePtr CreateSubgraphNode(
      const nnvm::Symbol &sym, const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_ngraph_subgraph_op");
    n->attrs.name = "_ngraph_subgraph_op" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    return n;
  }
  // Create a subgraph node based on a graph with inferred shapes, types
  // and storage types, then compile it with nGraph and store the
  // ngraph_bridge::Compiler object in NNVM's node attributes for execution.
  nnvm::NodePtr CreateSubgraphNode(
      const nnvm::Graph &sg, const int subgraph_id = 0) const override {
    nnvm::Symbol sym;
    sym.outputs = sg.outputs;
    auto n = CreateSubgraphNode(sym, subgraph_id);
    auto grad_req_map = GetAttr<std::vector<mxnet::OpReqType>>("grad_reqs");
    auto compiler = std::make_shared<ngraph_bridge::Compiler>(sg, grad_req_map);
    compiler->GetNgraph();
    n->attrs.parsed = compiler;
    return n;
  }
  // Create a Subgraph Selector with an embedded ngraph_bridge::Compiler for
  // nGraph support analysis
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    if (!compiler_) {
      auto &orig_graph = GetAttr<nnvm::Graph>("graph");
      auto grad_req_map = GetAttr<std::vector<mxnet::OpReqType>>("grad_reqs");
      compiler_ = std::make_shared<ngraph_bridge::Compiler>(orig_graph,
                                                            grad_req_map, true);
    }
    return std::make_shared<SgNgraphSelector>(compiler_.get());
  }

 private:
  mutable std::shared_ptr<ngraph_bridge::Compiler> compiler_;
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_NGRAPH

#endif  // MXNET_OPERATOR_CONTRIB_NGRAPH_INL_H_
