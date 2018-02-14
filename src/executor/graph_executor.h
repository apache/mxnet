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
 * \file graph_executor.h
 * \brief Executor to execute the computation graph.
 */
#ifndef MXNET_EXECUTOR_GRAPH_EXECUTOR_H_
#define MXNET_EXECUTOR_GRAPH_EXECUTOR_H_

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/imperative.h>
#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "./exec_pass.h"

namespace mxnet {

// forward declaration
namespace exec {
class GraphExecutor;
}

namespace exec {

using nnvm::Graph;

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v);

// graph executors
class GraphExecutor : public Executor {
 public:
  using Executor::MonitorCallback;

  GraphExecutor();
  virtual ~GraphExecutor();
  void Forward(bool is_train) override;
  void PartialForward(bool is_train, int step, int *step_left) override;
  void Backward(const std::vector<NDArray> &head_grads, bool is_train = true) override;
  const std::vector<NDArray>& outputs() const override;
  const std::unordered_map<std::string, NDArray>& in_arg_map() const override;
  const std::unordered_map<std::string, NDArray>& arg_grad_map() const override;
  const std::unordered_map<std::string, NDArray>& aux_state_map() const override;
  void Print(std::ostream &os) const override; // NOLINT(*)
  void SetMonitorCallback(const MonitorCallback& callback) override;
  // Initialize the rest of attributes
  // after setting up arguments.
  void FinishInitGraph(nnvm::Symbol symbol, nnvm::Graph g,
                       Executor* shared_exec = nullptr,
                       const nnvm::NodeEntryMap<NDArray>& feed_dict
                         = nnvm::NodeEntryMap<NDArray>());

  // initialize executor for bind
  void Init(nnvm::Symbol symbol,
            const Context& default_ctx,
            const std::map<std::string, Context>& ctx_map,
            const std::vector<NDArray>& in_args,
            const std::vector<NDArray>& arg_grad_store,
            const std::vector<OpReqType>& grad_req_types,
            const std::vector<NDArray>& aux_states,
            Executor* shared_exec = nullptr,
            const nnvm::NodeEntryMap<NDArray>& feed_dict
              = nnvm::NodeEntryMap<NDArray>());
  // initialize executor for simple bind
  void Init(nnvm::Symbol symbol,
            const Context& default_ctx,
            const std::map<std::string, Context>& ctx_map,
            const std::vector<Context>& in_arg_ctxes,
            const std::vector<Context>& arg_grad_ctxes,
            const std::vector<Context>& aux_state_ctxes,
            const std::unordered_map<std::string, TShape>& arg_shape_map,
            const std::unordered_map<std::string, int>& arg_dtype_map,
            const std::unordered_map<std::string, int>& arg_stype_map,
            const std::vector<OpReqType>& grad_req_types,
            const std::unordered_set<std::string>& shared_arg_names,
            std::vector<NDArray>* in_arg_vec,
            std::vector<NDArray>* arg_grad_vec,
            std::vector<NDArray>* aux_state_vec,
            std::unordered_map<std::string, NDArray>* shared_buffer = nullptr,
            Executor* shared_exec = nullptr,
            const nnvm::NodeEntryMap<NDArray>& feed_dict
              = nnvm::NodeEntryMap<NDArray>());

 protected:
  friend class mxnet::Imperative;
  // Information about operational node
  struct OpNode {
    // The name of the operator
    const char* opr_name;
    // the context of the node
    Context ctx;
    // The executor
    std::shared_ptr<OpExecutor> exec;
    // skip the execution of this node
    bool skip_exec_node{false};
    // cached operator handle
    Engine::OprHandle cached_opr{nullptr};
    // cached const vars, used for seg ops creation
    std::vector<Engine::VarHandle> use_vars;
    // cached mutate vars, used for seg ops creation
    std::vector<Engine::VarHandle> mutate_vars;
  };
  // a cached segment operator that executes a segment
  struct CachedSegOpr {
    // context of the operator
    Context ctx;
    // begin in topo order
    size_t topo_start;
    // end in topo order
    size_t topo_end;
    // the cached operator
    Engine::OprHandle opr = nullptr;
    // list of op executors
    std::vector<std::shared_ptr<OpExecutor> > exec_list;
  };
  // Initialize in_args, arg_grads, and aux_states
  void InitArguments(const nnvm::IndexedGraph& idx,
                     const nnvm::ShapeVector& inferred_shapes,
                     const nnvm::DTypeVector& inferred_dtypes,
                     const StorageTypeVector& inferred_stypes,
                     const std::vector<Context>& in_arg_ctxes,
                     const std::vector<Context>& arg_grad_ctxes,
                     const std::vector<Context>& aux_state_ctxes,
                     const std::vector<OpReqType>& grad_req_types,
                     std::vector<NDArray>* in_arg_vec,
                     std::vector<NDArray>* arg_grad_vec,
                     std::vector<NDArray>* aux_state_vec);
  // Initialize in_args, arg_grads and aux_states with
  // shared_buffer and shared_exec
  void InitArguments(const nnvm::IndexedGraph& idx,
                     const nnvm::ShapeVector& inferred_shapes,
                     const nnvm::DTypeVector& inferred_dtypes,
                     const StorageTypeVector& inferred_stypes,
                     const std::vector<Context>& in_arg_ctxes,
                     const std::vector<Context>& arg_grad_ctxes,
                     const std::vector<Context>& aux_state_ctxes,
                     const std::vector<OpReqType>& grad_req_types,
                     const std::unordered_set<std::string>& shared_arg_names,
                     const Executor* shared_exec,
                     std::unordered_map<std::string, NDArray>* shared_buffer,
                     std::vector<NDArray>* in_arg_vec,
                     std::vector<NDArray>* arg_grad_vec,
                     std::vector<NDArray>* aux_state_vec);
  // internal initialization of the graph for simple bind
  Graph InitGraph(nnvm::Symbol symbol,
                  const Context& default_ctx,
                  const std::map<std::string, Context>& ctx_map,
                  const std::vector<Context>& in_arg_ctxes,
                  const std::vector<Context>& arg_grad_ctxes,
                  const std::vector<Context>& aux_state_ctxes,
                  const std::vector<OpReqType>& grad_req_types);
  // intialize the full graph for simple bind, including gradient
  Graph InitFullGraph(nnvm::Symbol symbol,
                      const std::vector<OpReqType>& grad_req_types);
  // initialize the cached operator
  void InitCachedOps();
  // initialize the opr segments for bulk exec
  void InitOpSegs();
  // initialize the resources in the graph
  // initialize the memory of data entries
  // shared_pool: extra memory shared from other parts
  void InitDataEntryMemory(std::vector<NDArray>* shared_pool);
  // run ops from topo order start to end
  void RunOps(bool is_train, size_t topo_start, size_t topo_end);
  /*!
   * \brief Try to create a cached operator to run segments between start and end
   * \param topo_start beginning of segment
   * \param topo_end end of segment
   * \return the cached operator.
   *  ret.opr Can be nullptr if creation failed.
  */
  CachedSegOpr CreateCachedSegOpr(size_t topo_start, size_t topo_end);
  // run the monitor callback for node `nid`
  void ExecuteMonCallback(size_t nid);
  // peform bulking and segmentation on an inference graph
  void BulkInferenceOpSegs();
  // perform bulking and segmentation on a training graph
  void BulkTrainingOpSegs(size_t total_num_nodes);

  // internal graph
  nnvm::Graph graph_;
  // operator node
  std::vector<OpNode> op_nodes_;
  // internal data entry of each node
  std::vector<NDArray> data_entry_;
  // internal data pool of allocated entries.
  // these allocated entries can be used for static memory sharing between executors.
  std::vector<NDArray> data_pool_;
  // output arrays
  std::vector<NDArray> output_arrays_;
  // input argument map, key is arg name, value is arg's NDArray
  std::unordered_map<std::string, NDArray> in_arg_map_;
  // arg grad map, key is arg name, value is arg grad NDArray
  std::unordered_map<std::string, NDArray> arg_grad_map_;
  // aux state map, key is aux state name, value is aux state NDArray
  std::unordered_map<std::string, NDArray> aux_state_map_;
  // gradient store
  std::vector<std::pair<OpReqType, NDArray> > grad_store_;
  // array to hold head gradient.
  std::vector<NDArray> head_grad_array_;
  // entry to hold head gradient
  std::vector<nnvm::NodeEntry> head_grad_entry_;
  // the index map of entry to map.
  std::unordered_map<const nnvm::Node*, size_t> head_grad_map_;
  // number of outputs.
  size_t num_forward_outputs_{0};
  // number of inputs
  size_t num_forward_inputs_{0};
  // number of forward nodes
  size_t num_forward_nodes_{0};
  // saved operator for autograd
  std::unordered_map<const nnvm::Node*, OpStatePtr> saved_states_;
  // monitor call back
  std::function<void(const char*, void*)> monitor_callback_{nullptr};
  // whether to enable bulk execution
  bool prefer_bulk_execution_;
  // cached segment operator
  std::vector<CachedSegOpr> cached_seg_opr_;
  // verbose logging
  bool log_verbose_ = false;
};

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_H_
