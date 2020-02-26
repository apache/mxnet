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
#include <utility>
#include <tuple>

namespace mxnet {
namespace exec {

template <typename Attr>
using FAccessSubgraphAttr = std::function<std::tuple<const nnvm::ObjectPtr,
                                          std::vector<Attr>,
                                          std::vector<Attr>>
                              (const NodeAttrs& attrs)>;

using FAccessSubgraphShape = FAccessSubgraphAttr<mxnet::TShape>;
using FAccessSubgraphType = FAccessSubgraphAttr<int>;
using FAccessSubgraphStorageType = FAccessSubgraphAttr<int>;

template <typename Attr>
using FProvideSubgraphAttr = std::function<void (const NodeAttrs& attrs,
                                                 const std::vector<nnvm::ObjectPtr> &nodes,
                                                 const std::vector<std::vector<Attr>> &in_attrs,
                                                 const std::vector<std::vector<Attr>> &out_attrs)>;
using FProvideSubgraphShape = FProvideSubgraphAttr<mxnet::TShape>;
using FProvideSubgraphType = FProvideSubgraphAttr<int>;
using FProvideSubgraphStorageType = FProvideSubgraphAttr<int>;

using TIsFusion = bool;
using TIsFusionHelper = bool;

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
   *  This can be called multiple times if NDArray changed during reshape.
   *  It is safe to call it via an asynchronous engine lambda.
   */
  virtual void Setup() = 0;
  /*!
   * \brief run the operator given runtime context on device.
   *  This function call does not synchronize the stream.
   * \param rctx The runtime context passed in by environment.
   */
  virtual void Run(RunContext rctx, bool is_gpu) = 0;
  /*! \return the execution type */
  virtual ExecType exec_type() const = 0;
  /*! \return return engine variable for operator states */
  virtual engine::VarHandle var() const {
    return nullptr;
  }
  /*! \return return operator state */
  virtual OpStatePtr state() const {
    return OpStatePtr();
  }

  // TODO(alexzai): (MXNET-856) Remove instance member after subgraph feature added
 protected:
  std::vector<NDArray> in_array_fallback;
};

/*!
 * \brief per node vector of operator executors.
 * \note stored under attribute "op_exec"
 */
using OpExecVector = std::vector<std::shared_ptr<OpExecutor> >;

/*!
 * \brief per node vector of operator states.
 * \note stored under attribute "op_states"
 */
using OpStateVector = std::vector<OpStatePtr>;

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
 * \brief create OpExecutor for a node in graph
 *
 * \param g input graph
 * \param p_ret OpExecVector for input and output
 * \param p_state OpStateVector if it has.
 * \param i the id of the node
 */
void CreateOpExecs(const Graph& g, OpExecVector* p_ret, OpStateVector* p_state, size_t i);
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
 */
void AttachOpResources(const Graph& g);
/*!
 * \brief Attach Resource to the OpExecVector
 *
 * \param g input graph
 * \param op_execs OpExecutor vector
 * \param start_nid starting node id
 * \param end_nid end node id
 */
void AttachOpResources(const Graph& g,
                       const OpExecVector& op_execs,
                       size_t start_nid,
                       size_t end_nid);
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
 * \brief Eliminate common expressions in the graph.
 *
 * \param g input forward graph
 *
 * \return graph with common expressions eliminated
 */
Graph EliminateCommonExpr(Graph && g);

/*!
 * \brief Fuse pointwise operations in the forward pass.
 *
 * \param g input graph (needs to be entire graph, not just forward part)
 *
 * \return graph with fused pointwise operations in the forward pass
 */
Graph FusePointwiseForward(Graph&& g);

/*!
 * \brief Fuse pointwise operations in the backward pass.
 *
 * \param g input graph (needs to be entire graph, not just forward part)
 *
 * \return graph with fused pointwise operations in the backward pass
 */
Graph FusePointwiseBackward(Graph&& g);

/*!
 * \brief Issue a one-time warning that fusion is not possible for this platform or build.
 */
void WarnFusionNotSupported();

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
                 mxnet::ShapeVector&& shape_inputs = mxnet::ShapeVector(),
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

}  // namespace exec
}  // namespace mxnet

namespace nnvm {
namespace pass {
/*!
 * \brief Get the gradient graph whose outputs are gradients of xs wrt to ys.
 * \param graph The input graph.
 * \param ys The entries we want to take gradient from.
 * \param xs The input to take gradient with respect to.
 * \param ys_out_grad The symbol for additional gradient to be propagate back to y.
 * \param aggregate_fun Aggregation function applied to aggregate the inputs.
 * \param mirror_fun Optional mirror function to do mirror optimization and save memory.
 * \param attr_hint_fun Optional, hint function to output a node that like src, but its attr is same as like.
 * \param zero_ops Optional, list of operators that outputs a single zero array. The first one
 *  must be zeros_like.
 * \param copy_op_str Optional, name of the copy operation required to handle duplicates
 *  on the edge of the graph
 * \return A new graph, whose outputs correspond to inputs of xs.
 */
inline Graph MXGradient(
    Graph graph,
    std::vector<NodeEntry> ys,
    std::vector<NodeEntry> xs,
    std::vector<NodeEntry> ys_out_grad,
    std::function<NodeEntry(std::vector<NodeEntry>&& inputs)> aggregate_fun = nullptr,
    std::function<int(const Node& node)> mirror_fun = nullptr,
    std::function<NodeEntry(const NodeEntry& src, const NodeEntry &like)>
    attr_hint_fun = nullptr,
    std::vector<const Op*> zero_ops = std::vector<const Op*>(),
    std::string copy_op_str = std::string()) {
  graph.attrs["grad_ys"] = std::make_shared<any>(std::move(ys));
  graph.attrs["grad_xs"] = std::make_shared<any>(std::move(xs));
  graph.attrs["grad_ys_out_grad"] = std::make_shared<any>(std::move(ys_out_grad));
  if (aggregate_fun != nullptr) {
    graph.attrs["grad_aggregate_fun"] = std::make_shared<any>(aggregate_fun);
  }
  if (mirror_fun != nullptr) {
    graph.attrs["grad_mirror_fun"] = std::make_shared<any>(mirror_fun);
  }
  if (attr_hint_fun != nullptr) {
    graph.attrs["attr_hint_fun"] = std::make_shared<any>(attr_hint_fun);
  }
  if (zero_ops.size()) {
    graph.attrs["zero_ops"] = std::make_shared<any>(std::move(zero_ops));
  }
  if (copy_op_str != std::string()) {
      graph.attrs["copy_op"] = std::make_shared<any>(std::move(copy_op_str));
  }
  return ApplyPass(std::move(graph), "MXGradient");
}
}  // namespace pass
}  // namespace nnvm

#endif  // MXNET_EXECUTOR_EXEC_PASS_H_
