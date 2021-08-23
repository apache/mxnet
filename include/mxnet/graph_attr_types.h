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
 * \file graph_attr_types.h
 * \brief Data structures that can appear in graph attributes.
 */
#ifndef MXNET_GRAPH_ATTR_TYPES_H_
#define MXNET_GRAPH_ATTR_TYPES_H_

#include <mxnet/op_attr_types.h>
#include <vector>

namespace mxnet {

/*!
 * \brief The result holder of storage type of each NodeEntry in the graph.
 * \note Stored under graph.attrs["storage_type"], provided by Pass "InferStorageType"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "InferStorageType");
 *  const StorageVector& stypes = g.GetAttr<StorageTypeVector>("storage_type");
 *  // get storage type by entry id
 *  int entry_type = stypes[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 *
 * \sa FInferStorageType
 */
using StorageTypeVector = std::vector<int>;

/*!
+ * \brief The result holder of dispatch mode of each Node in the graph.
+ * \note Stored under graph.attrs["dispatch_mode"], provided by Pass "InferStorageType"
+ *
+ * \code
+ *  Graph g = ApplyPass(src_graph, "InferStorageType");
+ *  const DispatchModeVector& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
+ *  // get dispatch mode by entry node id
+ *  int node_type = dispatch_modes[nid];
+ * \endcode
+ *
+ * \sa FInferStorageType
+ */
using DispatchModeVector = std::vector<DispatchMode>;

}  // namespace mxnet

#endif  // MXNET_GRAPH_ATTR_TYPES_H_
