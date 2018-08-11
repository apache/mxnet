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

#ifndef MXNET_EXECUTOR_TRT_GRAPH_EXECUTOR_H_
#define MXNET_EXECUTOR_TRT_GRAPH_EXECUTOR_H_

#if MXNET_USE_TENSORRT

#include <map>
#include <string>
#include <vector>

#include "./graph_executor.h"

namespace mxnet {

namespace exec {

class TrtGraphExecutor : public GraphExecutor {
 public:
  static Executor* TensorRTBind(nnvm::Symbol symbol,
                                const Context& default_ctx,
                                const std::map<std::string, Context>& group2ctx,
                                std::vector<Context> *in_arg_ctxes,
                                std::vector<Context>* arg_grad_ctxes,
                                std::vector<Context>* aux_state_ctxes,
                                std::unordered_map<std::string, TShape>* arg_shape_map,
                                std::unordered_map<std::string, int>* arg_dtype_map,
                                std::unordered_map<std::string, int>* arg_stype_map,
                                std::vector<OpReqType>* grad_req_types,
                                const std::unordered_set<std::string>& param_names,
                                std::vector<NDArray>* in_args,
                                std::vector<NDArray>* arg_grads,
                                std::vector<NDArray>* aux_states,
                                std::unordered_map<std::string, NDArray>*
                                shared_data_arrays = nullptr,
                                Executor* shared_exec = nullptr);

  virtual void Init(nnvm::Symbol symbol,
                    const Context& default_ctx,
                    const std::map<std::string, Context>& ctx_map,
                    std::vector<Context> *in_arg_ctxes,
                    std::vector<Context> *arg_grad_ctxes,
                    std::vector<Context> *aux_state_ctxes,
                    std::unordered_map<std::string, TShape> *arg_shape_map,
                    std::unordered_map<std::string, int> *arg_dtype_map,
                    std::unordered_map<std::string, int> *arg_stype_map,
                    std::vector<OpReqType> *grad_req_types,
                    const std::unordered_set<std::string>& shared_arg_names,
                    std::vector<NDArray>* in_arg_vec,
                    std::vector<NDArray>* arg_grad_vec,
                    std::vector<NDArray>* aux_state_vec,
                    std::unordered_map<std::string, NDArray>* shared_buffer = nullptr,
                    Executor* shared_exec = nullptr,
                    const nnvm::NodeEntryMap<NDArray>& feed_dict
                      = nnvm::NodeEntryMap<NDArray>());

  // Returns symbol representing the TRT optimized graph for comparison purposes.
  nnvm::Symbol GetOptimizedSymbol();

 protected:
  Graph ReinitGraph(Graph&& g, const Context &default_ctx,
        const std::map<std::string, Context> &ctx_map,
        std::vector<Context> *in_arg_ctxes,
        std::vector<Context> *arg_grad_ctxes,
        std::vector<Context> *aux_state_ctxes,
        std::vector<OpReqType> *grad_req_types,
        std::unordered_map<std::string, TShape> *arg_shape_map,
        std::unordered_map<std::string, int> *arg_dtype_map,
        std::unordered_map<std::string, int> *arg_stype_map,
        std::unordered_map<std::string, NDArray> *params_map);

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
                     std::vector<NDArray>* aux_state_vec) override;
};

}  // namespace exec

}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT

#endif  // MXNET_EXECUTOR_TRT_GRAPH_EXECUTOR_H_
