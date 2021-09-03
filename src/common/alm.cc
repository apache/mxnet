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
 * Copyright (c) 2019 by Contributors
 * \file alm.cc
 * \brief Automatic Layout Manager
 * \author Dawid Tracz
 */

#include "alm.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "../operator/tensor/matrix_op-inl.h"

namespace mxnet {
namespace alm {

namespace {

nnvm::ObjectPtr CreateTransposeNode(const std::string& name, const alm::Transpose& axes) {
  nnvm::ObjectPtr newptr = nnvm::Node::Create();
  newptr->attrs.op = nnvm::Op::Get("transpose");
  newptr->attrs.name = name;
  // set tranpose axes
  std::ostringstream ss;
  ss << mxnet::TShape(axes.begin(), axes.end());
  newptr->attrs.dict["axes"] = ss.str();
  newptr->op()->attr_parser(&(newptr->attrs));
  return newptr;
}

std::unordered_map<std::string, mshadow::LayoutFlag> ConvertTargets(
    const std::unordered_map<std::string, std::string>& targets) {
  std::unordered_map<std::string, mshadow::LayoutFlag> ret;
  for (const auto& p : targets) {
    auto layout = mshadow::layoutFlag(p.second);
    if (layout == mshadow::kUNKNOWN) {
      LOG(WARNING) << "Unknown layout: " << p.second << ". ignoring...";
      continue;
    }
    ret.insert(std::make_pair(p.first, layout));
  }
  return ret;
}

}  // namespace

nnvm::Graph OptimizeLayout(nnvm::Graph&& g,
                           const std::unordered_map<std::string, std::string>& layout_targets) {
  std::unordered_map<std::string, mshadow::LayoutFlag> targets = ConvertTargets(layout_targets);
  static const auto& op_map = Op::GetAttr<mxnet::alm::FChangeLayout>("FChangeLayout");
  static const Op* transpose_op = Op::Get("transpose");
  std::unordered_set<nnvm::ObjectPtr> outputs;
  for (auto& o : g.outputs) outputs.insert(o.node);
  nnvm::NodeEntryMap<alm::Transpose> changed;
  struct ToDelete {
    nnvm::ObjectPtr node;  // output of the transpose
    size_t input_idx;
  };
  std::vector<ToDelete> to_delete;
  struct ToAdd {
    nnvm::ObjectPtr node;
    size_t input_idx;
    alm::Transpose axes;
  };
  std::vector<ToAdd> to_add;
  DFSVisit(
      g.outputs, [&targets, &outputs, &changed, &to_add, &to_delete](const nnvm::ObjectPtr& node) {
        std::vector<alm::Transpose> input_axes(node->inputs.size());
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          if (node->inputs[i].node->op() == transpose_op) {
            const auto& param = nnvm::get<op::TransposeParam>(node->inputs[i].node->attrs.parsed);
            if (IsIdentity(FromTShape(param.axes))) {
              to_delete.push_back({node, i});
              continue;
            }
          }
          auto it = changed.find(node->inputs[i]);
          if (it == changed.end()) continue;
          input_axes[i] = it->second;
        }
        auto fchange = op_map.get(node->op(), nullptr);
        if (fchange && outputs.count(node) == 0) {
          auto it = targets.find(node->op()->name);
          auto target_layout = it != targets.end() ? it->second : mshadow::kUNKNOWN;
          std::vector<alm::Transpose> output_axes;
          if (fchange(&node->attrs, target_layout, &input_axes, &output_axes))
            node->op()->attr_parser(&node->attrs);
          for (size_t i = 0; i < output_axes.size(); ++i) {
            if (IsIdentity(output_axes[i])) continue;
            changed.insert(std::make_pair(nnvm::NodeEntry(node, i, 0), output_axes[i]));
          }
        }
        for (size_t i = 0; i < input_axes.size(); ++i) {
          if (IsIdentity(input_axes[i])) continue;
          to_add.push_back({node, i, input_axes[i]});
        }
      });
  for (const auto& t : to_delete) {
    auto& tnode = t.node->inputs[t.input_idx].node;
    CHECK_EQ(tnode->inputs.size(), 1);
    t.node->inputs[t.input_idx] = tnode->inputs[0];
  }
  size_t node_no = 0;
  for (const auto& t : to_add) {
    auto tnode = CreateTransposeNode("ALM_transpose_" + std::to_string(node_no++), t.axes);
    tnode->inputs.push_back(t.node->inputs[t.input_idx]);
    t.node->inputs[t.input_idx] = nnvm::NodeEntry(tnode);
  }
  nnvm::Graph ret;
  ret.outputs = g.outputs;
  return ret;
}

Transpose Reverse(const Transpose& axes) {
  Transpose rev(axes.size());
  for (size_t i = 0; i < rev.size(); i++) rev[axes[i]] = i;
  return rev;
}

Transpose Compose(const Transpose& lhs, const Transpose& rhs) {
  if (lhs.empty()) return rhs;
  if (rhs.empty()) return lhs;
  CHECK_EQ(lhs.size(), rhs.size());
  Transpose ret(lhs.size());
  for (auto i = 0; i < ret.size(); ++i) ret[i] = lhs[rhs[i]];
  return ret;
}

bool IsIdentity(const Transpose& t) {
  for (size_t i = 0; i < t.size(); ++i) {
    if (t[i] != i) return false;
  }
  return true;
}

mshadow::LayoutFlag ApplyTranspose(mshadow::LayoutFlag layout, const Transpose& axes) {
  auto ret = mshadow::layoutFlag(ApplyTranspose(mshadow::toString(layout), axes));
  CHECK_NE(ret, mshadow::kUNKNOWN);
  return ret;
}

std::string ApplyTranspose(const std::string& layout, const Transpose& axes) {
  std::string ret(layout.size(), ' ');
  for (size_t i = 0; i < ret.size(); i++) ret[i] = layout[axes[i]];
  return ret;
}

Transpose FromTShape(const mxnet::TShape& s) {
  Transpose ret(s.ndim());
  std::copy(s.begin(), s.end(), ret.begin());
  return ret;
}

Transpose FactorCommonTranspose(std::vector<Transpose>* axes) {
  Transpose ret;
  for (auto& t : *axes) {
    if (IsIdentity(t)) continue;
    if (IsIdentity(ret)) {
      std::swap(t, ret);
      continue;
    }
    auto rev = Reverse(ret);
    t = Compose(t, rev);
  }
  return ret;
}

}  // namespace alm
}  // namespace mxnet
