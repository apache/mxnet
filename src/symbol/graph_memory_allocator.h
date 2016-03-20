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
 * \brief Memory pool holding a list of NDArrays for sharing between executors.
 */
struct GraphStoragePool {
  std::vector<NDArray> pool;
};

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
  /*! \brief dummy color for shared mem */
  static const uint32_t kDummyColor;
  /*! \brief constructor to the graph memory allocator */
  explicit GraphStorageAllocator(
      StaticGraph *graph,
      const std::vector<uint32_t>& topo_order,
      std::shared_ptr<GraphStoragePool> shared_mem) noexcept(false);
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
  /*! \brief shared memory pool */
  std::shared_ptr<GraphStoragePool> shared_mem_;
};
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_MEMORY_ALLOCATOR_H_
