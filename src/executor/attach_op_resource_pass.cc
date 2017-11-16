
/*!
 * Copyright (c) 2016 by Contributors
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
  auto& op_execs = nnvm::get<OpExecVector>(*g.attrs.at("op_execs"));
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& vdispatch = g.GetAttr<DispatchModeVector>("dispatch_mode");
  const auto& idx = g.indexed_graph();
  // Use global resource pool for each executor for now.
  std::map<Context, Resource> cached_temp;
  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    const Context &ctx = vctx[nid];
    auto& requested = op_execs[nid]->op_ctx.requested;
    requested.clear();
    const auto op = inode.source->op();
    if (fresource.count(op) != 0) {
      auto reqs = fresource[op](inode.source->attrs);
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
      CHECK(vdispatch[nid] != DispatchMode::kUndefined);
    }
    // extra resource requests for storage fallback
    if (vdispatch[nid] == DispatchMode::kFComputeFallback) {
      requested.push_back(ResourceManager::Get()->Request(ctx, ResourceRequest::kTempSpace));
    }
  }
  return g;
}
}  // namespace exec
}  // namespace mxnet
