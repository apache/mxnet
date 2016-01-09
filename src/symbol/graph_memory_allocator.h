/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_memory_allocator.h
 * \brief Memory allocator for graph executor.
*/
#ifndef MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
#define MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_

#include <mxnet/symbolic.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <algorithm>
#include "./static_graph.h"
#include "./graph_algorithm.h"
#include "../common/utils.h"

namespace mxnet {
/*!
 * \brief Memory allocators for the GraphExecutor.
 *  This class is intended to be used by GraphExecutor
 *  to allocate the memory for each DataEntryInfo.
 *
 *  The class algorithm works in two phase:
 *  (1) Planning Phase: GraphExecutor call Request and Release
 *      to request and release resources according to dependency.
 *      - Each call to Request will get a ResourceID that is used to
 *        identify the memory block assigned to each DataEntryInfo.
 *  (2) Allocating phase: GraphExecutor call InitMemory.
 *      - Then each DataEntry will call Get to get the real NDArray.
 *  (3) All the memory will be freed up when reference to all the related NDArray ends.
 */
class GraphStorageAllocator {
 public:
  /*! \brief resource index */
  typedef int64_t StorageID;
  /*! \brief bad storage id */
  static const StorageID kBadStorageID = -1;
  /*! \brief constructor to the graph memory allocator */
  explicit GraphStorageAllocator(
      StaticGraph *graph,
      const std::vector<uint32_t>& topo_order) noexcept(false);
  /*!
   * \brief Request a memory.
   * \param ctx the context of the graph
   * \param shape shape of the NDArray we want
   * \param node_id the node that is requesting the memory, used as hint.
   */
  StorageID Request(Context ctx, int type_flag, TShape shape, uint32_t node_id);
  /*!
   * \brief Release a memory.
   * \param id the storage ID of the memory.
   * \param node_id the node id in the graph that is releasing the memory.
   */
  void Release(StorageID id, uint32_t node_id);
  /*!
   * \brief Initialize all the memories requested
   * \return size of memory allocated.
   */
  size_t InitStorages();
  /*!
   * \brief Get the the memory allocated in planning phase.
   * \param id the storage id allocated in planning phase.
   * \param shape the shape of the NDArray requested.
   */
  NDArray Get(StorageID id, TShape shape);

 protected:
  /*! \brief internal storage entry */
  struct StorageEntry {
    /*! \brief id of the storage */
    StorageID id;
    /*! \brief the context of the storage */
    Context ctx;
    /*! \brief the data type enum of the storage */
    int type_flag;
    /*! \brief maximum size of the storage that is requested */
    size_t max_size;
    /*! \brief node index that released it last time */
    uint32_t released_by_node;
    /*! \brief the actual NDArray to hold the data */
    NDArray data;
    /*! \brief constructor */
    StorageEntry() : max_size(0), released_by_node(0) {}
  };
  /*!
   * \brief Allocate a StorageID when Request cannot found existing ones.
   * \param ctx the context of the graph
   * \param shape shape of the NDArray we want
   */
  StorageID Alloc(Context ctx, int type_flag, size_t size);
  /*!
   * \brief Initialize the colors of graph nodes.
   * \param topo_order the topological order in the graph.
   */
  void InitColor(const std::vector<uint32_t> &topo_order);
  /*! \brief reference to the computation graph */
  StaticGraph *graph_;
  /*! \brief all the resources available */
  std::vector<std::unique_ptr<StorageEntry> > data_;
  /*! \brief scale used for rough match */
  size_t match_range_;
  /*!
   * \brief free list of storage entries, maps size to free list
   */
  std::multimap<size_t, StorageEntry*> free_;
  /*!
   * \brief color of nodes in the graph, used for auxiliary policy making.
  */
  std::vector<uint32_t> node_color_;
  /*! \brief whether use color based match algorithm */
  uint32_t num_match_color_;
};

// put implementation in header files for now
GraphStorageAllocator::GraphStorageAllocator(
    StaticGraph *graph,
    const std::vector<uint32_t>& topo_order) noexcept(false)
    : graph_(graph) , num_match_color_(0) {
  match_range_ = dmlc::GetEnv("MXNET_EXEC_MATCH_RANGE", 16);
  // if we set this to 1, this means no color based match.
  // color based match will cost a bit more memory usually
  // but also enables more parallelization.
  num_match_color_ = static_cast<uint32_t>(common::GetExecNumMatchColor());
  this->InitColor(topo_order);
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
    if (node_color_[e->released_by_node] != node_color_[node_id]) continue;
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
    if (node_color_[e->released_by_node] != node_color_[node_id]) continue;
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
    TShape shape = mshadow::Shape1(e->max_size);
    e->data = NDArray(shape, e->ctx, false, e->type_flag);
    total += e->max_size * mshadow::mshadow_sizeof(e->type_flag);
  }
  return total;
}

NDArray GraphStorageAllocator::Get(StorageID id, TShape shape) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  return e->data.Slice(0, shape.Size()).Reshape(shape);
}
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
