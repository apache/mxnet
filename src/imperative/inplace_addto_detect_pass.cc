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
 * Copyright (c) 2016 by Contributors
 * \file inplace_addto_detect_pass.cc
 * \brief Detect whether inplace addto operation is possible for certain op.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

#include "./exec_pass.h"

namespace mxnet {
namespace exec {

Graph DetectInplaceAddTo(Graph g) {
  nnvm::StorageVector storage_id =
      g.MoveCopyAttr<nnvm::StorageVector>("storage_id");
  std::vector<int> storage_inplace_index =
      g.MoveCopyAttr<std::vector<int> >("storage_inplace_index");
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  auto& idx = g.indexed_graph();
  // reference cont.
  std::vector<int> ref_count(idx.num_node_entries(), 0);
  std::vector<int> addto_entry(idx.num_node_entries(), 0);
  std::vector<int> skip_plus_node(idx.num_nodes(), 0);

  for (auto& e : idx.outputs()) {
    ++ref_count[idx.entry_id(e)];
  }
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    for (auto &e : idx[nid].inputs) {
      ++ref_count[idx.entry_id(e)];
    }
  }

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->op() != ewise_plus_op) continue;
    int sid = storage_id[idx.entry_id(inode.inputs[0])];
    if (sid != storage_id[idx.entry_id(nid, 0)]) continue;
    if (idx[inode.inputs[0].node_id].source->is_variable()) continue;
    if (idx[inode.inputs[1].node_id].source->is_variable()) continue;
    uint32_t eid_rhs  = idx.entry_id(inode.inputs[1]);
    if (ref_count[eid_rhs] != 1) continue;
    if (inode.inputs[0].node_id >= inode.inputs[1].node_id) continue;
    // TODO(haibin) support inplace addto for Dynamic Storage
    if (storage_id[eid_rhs] == kDynamicStorageID) continue;
    CHECK_NE(storage_id[eid_rhs], sid);
    storage_id[eid_rhs] = sid;
    addto_entry[eid_rhs] = 1;
    storage_inplace_index[eid_rhs] = -1;
    skip_plus_node[nid] = 1;
  }

  g.attrs["storage_id"] = std::make_shared<nnvm::any>(std::move(storage_id));
  g.attrs["storage_inplace_index"] = std::make_shared<nnvm::any>(
      std::move(storage_inplace_index));
  g.attrs["addto_entry"] = std::make_shared<nnvm::any>(std::move(addto_entry));
  g.attrs["skip_plus_node"] = std::make_shared<nnvm::any>(std::move(skip_plus_node));
  return g;
}

}  // namespace exec
}  // namespace mxnet
