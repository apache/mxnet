
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
  auto& fresource =
      nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
  auto& op_execs = nnvm::get<OpExecVector>(*g.attrs.at("op_execs"));
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& idx = g.indexed_graph();

  std::map<Context, int> tspace_cnt;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (fresource.count(inode.source->op()) == 0) continue;
    auto reqs = fresource[inode.source->op()](inode.source->attrs);
    int cnt = 0;
    for (const ResourceRequest& req : reqs) {
      if (req.type == ResourceRequest::kTempSpace) {
        cnt++;
      }
    }
    const Context &ctx = vctx[nid];
    if (tspace_cnt.count(ctx)) {
      tspace_cnt.at(ctx) = std::max(tspace_cnt.at(ctx), cnt);
    } else {
      tspace_cnt[ctx] = cnt;
    }
  }

  std::map<Context, std::vector<Resource>> cached_tspace;
  for (const auto& kv: tspace_cnt) {
    for (int i = 0; i < kv.second; ++i) {
      Resource r = ResourceManager::Get()->Request(kv.first, ResourceRequest::kTempSpace);
      cached_tspace[kv.first].push_back(r);
    }
  }

  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (fresource.count(inode.source->op()) == 0) continue;
    auto reqs = fresource[inode.source->op()](inode.source->attrs);
    auto& requested = op_execs[nid]->op_ctx.requested;
    requested.clear();
    // Get the resource of temporal space.
    int tspace_idx = 0;
    for (const ResourceRequest& req : reqs) {
      const Context &ctx = vctx[nid];
      if (req.type == ResourceRequest::kTempSpace) {
        requested.push_back(cached_tspace.at(ctx)[tspace_idx++]);
      } else if (req.type == ResourceRequest::kRandom) {
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
      } else {
        LOG(FATAL) << "resource type not yet supported";
      }
    }
  }
  return g;
}
}  // namespace exec
}  // namespace mxnet
