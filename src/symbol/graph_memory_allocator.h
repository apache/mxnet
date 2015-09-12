/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_memory_allocator.h
 * \brief Memory allocator for graph executor.
*/
#ifndef MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
#define MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_

#include <mxnet/symbolic.h>
#include <mxnet/narray.h>
#include <map>
#include <vector>

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
 *      - Then each DataEntry will call Get to get the real NArray.
 *  (3) All the memory will be freed up when reference to all the related NArray ends.
 */
class GraphStorageAllocator {
 public:
  /*! \brief resource index */
  typedef int64_t StorageID;
  /*! \brief bad storage id */
  static const StorageID kBadStorageID = -1;
  /*! \brief constructor to the graph memory allocator */
  explicit GraphStorageAllocator(StaticGraph *graph);
  /*!
   * \brief Request a memory.
   * \param ctx the context of the graph
   * \param shape shape of the NArray we want
   * \param node_id the node that is requesting the memory, used as hint.
   */
  StorageID Request(Context ctx, TShape shape, uint32_t node_id);
  /*!
   * \brief Release a memory.
   * \param id the storage ID of the memory.
   * \param node_id the node id in the graph that is releasing the memory.
   */
  void Release(StorageID id, uint32_t node_id);
  /*! \brief Initialize all the memories requested */
  void InitStorages();
  /*!
   * \brief Get the the memory allocated in planning phase.
   * \param id the storage id allocated in planning phase.
   * \param shape the shape of the NArray requested.
   */
  NArray Get(StorageID id, TShape shape);

 protected:
  /*! \brief internal storage entry */
  struct StorageEntry {
    /*! \brief id of the storage */
    StorageID id;
    /*! \brief the context of the storage */
    Context ctx;
    /*! \brief maximum size of the storage that is requested */
    size_t max_size;
    /*! \brief the actual NArray to hold the data */
    NArray data;
    /*! \brief constructor */
    StorageEntry() : max_size(0) {}
  };
  /*!
   * \brief Allocate a StorageID when Request cannot found existing ones.
   * \param ctx the context of the graph
   * \param shape shape of the NArray we want
   */
  StorageID Alloc(Context ctx, size_t size);

  /*! \brief reference to the computation graph */
  StaticGraph *graph_;
  /*! \brief all the resources available */
  std::vector<std::unique_ptr<StorageEntry> > data_;
  /*!
   * \brief free list of storage entries, maps size to free list
   */
  std::multimap<size_t, StorageEntry*> free_;
};

// put implementation in header files for now
GraphStorageAllocator::GraphStorageAllocator(StaticGraph *graph)
    : graph_(graph) {}

GraphStorageAllocator::StorageID
GraphStorageAllocator::Alloc(Context ctx, size_t size) {
  StorageID id = static_cast<StorageID>(data_.size());
  std::unique_ptr<StorageEntry> ptr(new StorageEntry());
  ptr->id = id;
  ptr->ctx = ctx;
  ptr->max_size = size;
  data_.push_back(std::move(ptr));
  return id;
}

GraphStorageAllocator::StorageID
GraphStorageAllocator::Request(Context ctx, TShape shape, uint32_t node_id) {
  size_t size = shape.Size();
  auto begin = free_.lower_bound(size);
  auto end = free_.upper_bound(size);
  // vector of possible candidates
  for (auto it = begin; it != end; ++it) {
    StorageEntry *e = it->second;
    if (e->ctx != ctx) continue;
    // Use exect matching strategy
    // TODO(bing): think of other strategies, for example, rough match.
    if (e->max_size != size) continue;
    // find a exact match, erase from map and return
    free_.erase(it);
    return e->id;
  }
  // cannot find anything return a new one.
  return this->Alloc(ctx, size);
}

void GraphStorageAllocator::Release(StorageID id, uint32_t node_id) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  free_.insert({e->max_size, e});
}

void GraphStorageAllocator::InitStorages() {
  for (size_t i = 0; i < data_.size(); ++i) {
    StorageEntry *e = data_[i].get();
    TShape shape = mshadow::Shape1(e->max_size);
    e->data = NArray(shape, e->ctx);
  }
}

NArray GraphStorageAllocator::Get(StorageID id, TShape shape) {
  CHECK_NE(id, kBadStorageID);
  StorageEntry *e = data_[id].get();
  return e->data.Slice(0, shape.Size()).Reshape(shape);
}
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
