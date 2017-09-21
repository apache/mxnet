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
 * \file attach_op_resource_pass.cc
 * \brief Pass to attach resource to OpExecVector of the graph.
 */
#include <mxnet/resource.h>
#include <mxnet/op_attr_types.h>
#include "./exec_pass.h"

namespace mxnet {
namespace exec {

Graph AttachOpResources(Graph g) {
  static auto& fresource =
      nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
  static auto& fmutate = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  auto& op_execs = nnvm::get<OpExecVector>(*g.attrs.at("op_execs"));
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& vdispatch = g.GetAttr<DispatchTypeVector>("dispatch_type");
  const auto& vstype = g.GetAttr<StorageTypeVector>("storage_type");
  const auto& idx = g.indexed_graph();
  // Use global resource pool for each executor for now.
  std::map<Context, Resource> cached_temp;
  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    const auto dispatch_type = vdispatch[nid];
    if (inode.source->is_variable()) continue;
    const Context &ctx = vctx[nid];
    auto& requested = op_execs[nid]->op_ctx.requested;
    const auto op = inode.source->op();
    const auto op_attrs = inode.source->attrs;
    if (fresource.count(op) != 0) {
      auto reqs = fresource[op](op_attrs);
      requested.clear();
      // Get the resource of temporal space.
      for (const ResourceRequest& req : reqs) {
        if (req.type == ResourceRequest::kTempSpace) {
          if (cached_temp.count(ctx) != 0) {
            requested.push_back(cached_temp.at(ctx));
          } else {
            Resource r = ResourceManager::Get()->Request(ctx, req);
            requested.push_back(r);
            cached_temp[ctx] = r;
          }
        } else if (req.type == ResourceRequest::kRandom) {
          requested.push_back(ResourceManager::Get()->Request(ctx, req));
        } else {
          LOG(FATAL) << "resource type not yet supported";
        }
      }
      CHECK_NE(vdispatch[nid], kDispatchUndefined);
    }
    // extra resource requests for storage fallback
    if (vdispatch[nid] == kDispatchFComputeFallback) {
      auto req = ResourceRequest::kTempSpace;
      // resource for inputs
      for (const auto& e : inode.inputs) {
        const auto eid = idx.entry_id(e);
        CHECK_NE(vstype[eid], kUndefinedStorage);
        if (vstype[eid] != kDefaultStorage) {
          requested.push_back(ResourceManager::Get()->Request(ctx, req));
        }
      }
      // resource for outputs
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        CHECK_NE(vstype[eid], kUndefinedStorage);
        if (vstype[eid] != kDefaultStorage) {
          requested.push_back(ResourceManager::Get()->Request(ctx, req));
        }
      }
      // resource for mutatable inputs
      if (fmutate.count(op)) {
        const auto mutate_idx = fmutate[op](op_attrs);
        for (const auto i : mutate_idx) {
          uint32_t eid = idx.entry_id(inode.inputs[i]);
          if (vstype[eid] != kDefaultStorage) {
            requested.push_back(ResourceManager::Get()->Request(ctx, req));
          }
        }
      }
    }
  }
  return g;
}
}  // namespace exec
}  // namespace mxnet
