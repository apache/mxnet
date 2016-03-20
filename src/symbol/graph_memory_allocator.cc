/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_memory_allocator.cc
 * \brief Memory allocator for graph executor.
*/
#include "graph_memory_allocator.h"

namespace mxnet {
const uint32_t GraphStorageAllocator::kDummyColor = 1 << 31;

GraphStorageAllocator::GraphStorageAllocator(
    StaticGraph *graph,
    const std::vector<uint32_t>& topo_order,
    std::shared_ptr<GraphStoragePool> shared_mem) noexcept(false)
    : graph_(graph) , num_match_color_(0), shared_mem_(shared_mem) {
  match_range_ = dmlc::GetEnv("MXNET_EXEC_MATCH_RANGE", 16);
  // if we set this to 1, this means no color based match.
  // color based match will cost a bit more memory usually
  // but also enables more parallelization.
  num_match_color_ = static_cast<uint32_t>(common::GetExecNumMatchColor());
  this->InitColor(topo_order);

  for (auto& it : shared_mem_->pool) {
    CHECK(!it.is_none());
    CHECK_EQ(it.shape().ndim(), 1);
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->ctx = it.ctx();
    ptr->type_flag = it.dtype();
    ptr->max_size = it.shape()[0];
    ptr->data = it;
    data_.push_back(std::move(ptr));

    StorageEntry *e = data_[id].get();
    // set to dummy node.
    e->released_by_node = node_color_.size() - 1;
    free_.insert({e->max_size, e});
  }
}

void GraphStorageAllocator::InitColor(const std::vector<uint32_t>& topo_order) {
  std::vector<uint32_t> importance(graph_->nodes.size(), 0);
  for (size_t i = 0; i < topo_order.size(); ++i) {
    uint32_t nid = topo_order[i];
    if (graph_->nodes[nid].is_variable()) continue;
    importance[nid] = 1;
  }
  num_match_color_ = graph::ColorNodeGroup(
      *graph_, topo_order,
      importance, num_match_color_,
      &node_color_);
  //  dummy color for shared memory
  node_color_.push_back(kDummyColor);
}

GraphStorageAllocator::StorageID
GraphStorageAllocator::Alloc(Context ctx, int type_flag, size_t size) {
  StorageID id = static_cast<StorageID>(data_.size());
  std::unique_ptr<StorageEntry> ptr(new StorageEntry());
  ptr->id = id;
  ptr->ctx = ctx;
  ptr->type_flag = type_flag;
  ptr->max_size = size;
  data_.push_back(std::move(ptr));
  return id;
}

GraphStorageAllocator::StorageID
GraphStorageAllocator::Request(Context ctx, int type_flag, TShape shape, uint32_t node_id) {
  // search memory block in [size / match_range_, size * match_range_)
  size_t size = shape.Size();
  if (match_range_ == 0) return this->Alloc(ctx, type_flag, size);
  auto begin = free_.lower_bound(size / match_range_);
  auto mid = free_.lower_bound(size);
  auto end = free_.upper_bound(size * match_range_);
  // TODO(bing, min) consider better strategy
  // search for memory blocks larger than requested
  for (auto it = mid; it != end; ++it) {
    StorageEntry *e = it->second;
    if (e->ctx != ctx) continue;
    if (e->type_flag != type_flag) continue;
    if (node_color_[e->released_by_node] != kDummyColor
        && node_color_[e->released_by_node] != node_color_[node_id]) continue;
    if (!e->data.is_none() && size > e->max_size) continue;
    // Use exect matching strategy
    e->max_size = std::max(size, e->max_size);
    // find a exact match, erase from map and return
    free_.erase(it);
    return e->id;
  }
  // then search for memory blocks smaller than requested space
  for (auto it = mid; it != begin;) {
    --it;
    StorageEntry *e = it->second;
    if (e->ctx != ctx) continue;
    if (e->type_flag != type_flag) continue;
    if (node_color_[e->released_by_node] != kDummyColor
        && node_color_[e->released_by_node] != node_color_[node_id]) continue;
    if (!e->data.is_none() && size > e->max_size) continue;
    // Use exect matching strategy
    e->max_size = std::max(size, e->max_size);
    // find a exact match, erase from map and return
    free_.erase(it);
    return e->id;
  }
  // cannot find anything return a new one.
  return this->Alloc(ctx, type_flag, size);
}

void GraphStorageAllocator::Release(StorageID id, uint32_t node_id) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  e->released_by_node = node_id;
  free_.insert({e->max_size, e});
}

size_t GraphStorageAllocator::InitStorages() {
  size_t total = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    StorageEntry *e = data_[i].get();
    if (e->data.is_none()) {
      TShape shape = mshadow::Shape1(e->max_size);
      e->data = NDArray(shape, e->ctx, false, e->type_flag);
      total += e->max_size * mshadow::mshadow_sizeof(e->type_flag);
      shared_mem_->pool.push_back(e->data);
    }
  }
  CHECK_EQ(shared_mem_->pool.size(), data_.size());
  return total;
}

NDArray GraphStorageAllocator::Get(StorageID id, TShape shape) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  return e->data.Slice(0, shape.Size()).Reshape(shape);
}
}  // namespace mxnet
