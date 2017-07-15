/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph_attr_types.h
 * \brief Data structures that can appear in graph attributes.
 */
#ifndef MXNET_GRAPH_ATTR_TYPES_H_
#define MXNET_GRAPH_ATTR_TYPES_H_

#include <vector>

namespace mxnet {

/*!
 * \brief The result holder of storage type of each NodeEntry in the graph.
 * \note Stored under graph.attrs["storage_type"], provided by Pass "InferStorageType"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "InferStorageType");
 *  const StorageVector& stypes = g.GetAttr<StorageTypeVector>("storage_type");
 *  // get shape by entry id
 *  int entry_type = stypes[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 *
 * \sa FInferStorageType
 */
using StorageTypeVector = std::vector<int>;

}  // namespace mxnet

#endif  // MXNET_GRAPH_ATTR_TYPES_H_
