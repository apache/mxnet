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
 * \file exec_pass.h
 * \brief All the execution related pass and data structures.
 */
#ifndef MXNET_EXECUTOR_EXEC_PASS_H_
#define MXNET_EXECUTOR_EXEC_PASS_H_

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/graph_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <vector>
#include <memory>
#include <string>

namespace mxnet {
namespace exec {

/*! \brief reuse graph definition */
using nnvm::Graph;

const int kBadStorageID = -1;
const int kExternalStorageID = -2;
const int kDynamicStorageID = -3;

const int kNonDefaultStorage = -2;

/*!
 * \brief executor to execute an operator
 * This is a graph executor dependent interface
 * that unifies all the operator
 */
class OpExecutor {
 public:
  /*! \brief input data arrays, which may be either input or aux */
  std::vector<NDArray> in_array;
  /*! \brief output data arrays */
  std::vector<NDArray> out_array;
  /*! \brief output requirement on each array */
  std::vector<OpReqType> req;
  /*! \brief runtime op context, contains allocated resources */
  OpContext op_ctx;
  /*! \brief virtual destructor */
  virtual ~OpExecutor() {}
  /*!
   * \brief Setup the executor for given NDArray member
   * this can be called multiple times if NDArray changed during reshape.
   *  It is safe to call it via asynchronize engine lambda
   */
  virtual void Setup() = 0;
  /*!
   * \brief run the operator given runtime context on device.
   *  This function call do not synchronize the stream.
   * \param rctx The runtime context passed in by environment.
   */
  virtual void Run(RunContext rctx, bool is_gpu) = 0;
  /*! \return the execution type */
  virtual ExecType exec_type() const = 0;
  /*! \return return engine variable for operator states */
  virtual engine::VarHandle var() const {
    return nullptr;
  }
};

/*!
 * \brief per node vector of operator executors.
 * \note stored under attribute "op_exec"
 */
using OpExecVector = std::vector<std::shared_ptr<OpExecutor> >;

/*!
 * \brief per node context vector
 * \node stored under "context"
 */
using ContextVector = std::vector<Context>;

/*!
 * \brief per node device mask vector
 * \node stored under "dev_mask"
 */
using DevMaskVector = std::vector<int>;

/*!
 * \brief Attach OpExecutor to the graph attributes.
 *
 * \param g input graph
 * \return graph with new attribute "op_exec" of type OpExecVector
 *  The fields on the OpExecVector are not yet been setup.
 */
Graph AttachOpExecs(Graph g);

/*!
 * \brief Attach Resource to the OpExecVector of the graph.
 *
 * \param g input graph need to contain op_exec attribute.
 *
 * \return graph with new attribute "op_exec" of type OpExecVector
 *  The fields on the OpExecVector are not yet been setup.
 */
Graph AttachOpResources(Graph g);

/*!
 * \brief Discover chance of inplace addto operators.
 *  i.e. z = plus(z, source_op), and encourage it to become z += source_op.
 *
 * This optimization is coupled with executor. This is helpful to reduce memory
 * and computation for gradient aggregation of RNN.
 *
 * Require storage placement to be already finished.
 *
 * \param g input graph need to contain op_exec attribute.
 *
 * \return graph two new attributes, changes attribute "storage_id".
 *  - "addto_entry", std::vector<bool> size=g.num_node_entries()
 *    - addto_entry[eid] == 1, the corresponding op need to be performed using req=kAddTo
 *  - "skip_plus_node", std::vector<int> if set to 1, current op's execution is skiped.
 */
Graph DetectInplaceAddTo(Graph g);

/*!
 * \brief Infer shapes in the graph given the information.
 * \param graph The input graph.
 * \param shape_inputs The shapes of input symbols to the graph.
 * \param shape_attr_key The key to the node attribute that can indicate shape. This is
 *                       the place where manual hint for shapes could be injected.
 * \return A graph with new attribute "shape" containing inferred shape of each NodeEntry.
 *         The index of ShapeVector is given by graph.indexed_graph().entry_id.
 */
Graph InferShape(Graph&& graph,
                 nnvm::ShapeVector&& shape_inputs = nnvm::ShapeVector(),
                 const std::string& shape_attr_key = "");

/*!
 * \brief Infer types in the graph given the information.
 * \param graph The input graph.
 * \param dtype_inputs The types of input symbols to the graph.
 * \param dtype_attr_key The key to the node attribute that can indicate types. This is
 *                       the place where manual hint for types could be injected.
 * \return A graph with new attribute "dtype" containing inferred type of each NodeEntry.
 *         The index of ShapeVector is given by graph.indexed_graph().entry_id.
 */
Graph InferType(Graph&& graph,
                nnvm::DTypeVector&& dtype_inputs = nnvm::DTypeVector(),
                const std::string& dtype_attr_key = "");

/*!
 * \brief Infer storage types in the graph given the information.
 * \param graph The input graph.
 * \param storage_type_inputs The storage types of input symbols to the graph.
 * \param storage_type_attr_key The key to the node attribute that can indicate storage types.
                                This is the place where manual hint for types could be injected.
 * \return A graph with new attribute "storage_type" containing inferred type of each NodeEntry.
 *         The index of StorageTypeVector is given by graph.indexed_graph().entry_id.
 */
Graph InferStorageType(Graph&& graph,
                       StorageTypeVector&& storage_type_inputs = StorageTypeVector(),
                       const std::string& storage_type_attr_key = "");

/*! \brief The default storage type inference function, which assigns all undefined
 *         storage types to kDefaultStorage. If all of input and output storage types
 *         are kDefaultStorage, DispatchMode::kFCompute is assigned to dispatch_mode. Otherwise,
 *         DispatchMode::kFComputeFallback is assigned to dispatch_mode.
 */
bool DefaultStorageType(const nnvm::NodeAttrs& attrs,
                        const int dev_mask,
                        DispatchMode* dispatch_mode,
                        std::vector<int> *iattr,
                        std::vector<int> *oattr);

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_EXEC_PASS_H_
