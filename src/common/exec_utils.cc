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
 * \file exec_utils.cc
 * \brief Implementation of executor util functions.
 */

#include "exec_utils.h"
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace mxnet {
namespace common {

void CopyGraph(nnvm::Graph *dst, const nnvm::Graph &src, bool copy_variables) {
  using nnvm::Node;
  using nnvm::ObjectPtr;
  using nnvm::NodeEntry;
  std::unordered_map<Node*, ObjectPtr> old_new;
  // use DFSVisit to copy all the nodes
  DFSVisit(src.outputs, [&old_new, copy_variables](const ObjectPtr& node) {
      ObjectPtr np;
      if (copy_variables || !node->is_variable()) {
        np = Node::Create();
        np->attrs = node->attrs;
      } else {
        np = node;
      }
      old_new[node.get()] = std::move(np);
    });
  // connect nodes of new graph
  for (const auto &kv : old_new) {
    for (const NodeEntry& e : kv.first->inputs) {
      Node *ptr = e.node.get();
      kv.second->inputs.emplace_back(NodeEntry{old_new[ptr], e.index, e.version});
    }
    for (const ObjectPtr& p : kv.first->control_deps) {
      kv.second->control_deps.emplace_back(old_new[p.get()]);
    }
  }
  // set the head
  for (const NodeEntry &e : src.outputs) {
    (*dst).outputs.emplace_back(NodeEntry{old_new[e.node.get()], e.index, e.version});
  }
}

bool CheckForInputNameDuplicates(const nnvm::IndexedGraph &idx) {
  std::unordered_set<std::string> names;
  for (const auto& nid : idx.input_nodes()) {
    const std::string &name = idx[nid].source->attrs.name;
    if (names.count(name)) {
      LOG(WARNING) << "Variable name " << name << " is used more than once!";
      return false;
    }
    names.insert(name);
  }
  return true;
}

}  // namespace common
}  // namespace mxnet
