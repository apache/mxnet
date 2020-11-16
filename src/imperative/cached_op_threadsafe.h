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

// Threadsafe and minimal functionality cached op version for Inference
// lot of code reused from cached_op.h
#ifndef MXNET_IMPERATIVE_CACHED_OP_THREADSAFE_H_
#define MXNET_IMPERATIVE_CACHED_OP_THREADSAFE_H_

#include <mxnet/imperative.h>
#include <vector>
#include <atomic>
#include <utility>
#include <string>
#include <unordered_map>
#include "./cached_op.h"



namespace mxnet {
/*! \brief CachedOp Parameters*/
struct CachedOpThreadSafeConfig
    : public dmlc::Parameter<CachedOpThreadSafeConfig> {
  // keeping the config minimal
  // inlining, bulking, dynamic shapes, static allocing and shaping not
  // supported
  // data_indices indicates which of the indices from the arguments are data
  mxnet::Tuple<uint32_t> data_indices;
  // param_indices indicates which of the indices from the arguments are params
  mxnet::Tuple<uint32_t> param_indices;
  // decides the bulk size for dynamic forward
  uint32_t forward_bulk_size;
  bool static_alloc;
  bool static_shape;
  DMLC_DECLARE_PARAMETER(CachedOpThreadSafeConfig) {
    DMLC_DECLARE_FIELD(static_alloc)
    .set_default(false)
    .describe("Statically allocate memory to improve speed. "
              "Memory usage may increase.");
    DMLC_DECLARE_FIELD(static_shape)
    .set_default(false)
    .describe("Optimize for invariant input shapes between iterations. "
              "Must also set static_alloc to True. "
              "Change of input shapes is still allowed but slower.");
    DMLC_DECLARE_FIELD(forward_bulk_size)
     .set_default(Imperative::BulkExecMaxNodeTrainFwd())
     .describe("Segment size of bulk execution during dynamic forward");
    DMLC_DECLARE_FIELD(data_indices)
        .set_default(mxnet::Tuple<uint32_t>())
        .describe("Position of argument variables.");
            DMLC_DECLARE_FIELD(param_indices)
        .set_default(mxnet::Tuple<uint32_t>())
        .describe("Position of parameters.");
  }
};

// Thread local buff to store internal states of the graph
// Used in dynamic_forward
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local std::vector<NDArray> buff;
#else
    static MX_THREAD_LOCAL std::vector<NDArray> buff;
#endif



class CachedOpThreadSafe : public CachedOp {
 public:
  CachedOpThreadSafe(
      const nnvm::Symbol &sym,
      const std::vector<std::pair<std::string, std::string>> &flags);
  ~CachedOpThreadSafe();
  uint32_t num_inputs() const {
      return fwd_graph_.indexed_graph().input_nodes().size();
  }
  uint32_t num_outputs() const {
      return fwd_graph_.outputs.size();
  }
  const std::unordered_set<uint32_t>& mutable_input_nodes() const {
    return fwd_graph_.indexed_graph().mutable_input_nodes();
  }
  OpStatePtr Forward(
      const std::shared_ptr<CachedOp>& op_ptr,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs,
      const Context& default_ctx);
  std::vector<std::string> ListForwardInputNames() const {
    nnvm::Symbol sym = GetForwardSym();
    return sym.ListInputNames(nnvm::Symbol::kAll);
  }
  std::vector<std::string> ListForwardOutputNames() const {
    nnvm::Symbol sym = GetForwardSym();
    return sym.ListOutputNames();
  }
  nnvm::Symbol GetForwardSym() const {
    nnvm::Symbol sym;
    sym.outputs = fwd_graph_.outputs;
    return sym;
  }

  struct GraphInfo;
 private:
  struct DynamicRuntime;

  OpStatePtr GetCachedOpState(const Context& ctx);

  OpStatePtr DynamicForward(const Context& default_ctx,
                            const std::vector<NDArray*>& inputs,
                            const std::vector<NDArray*>& outputs);

  CachedOpThreadSafeConfig config_;
  nnvm::Graph fwd_graph_;
  std::mutex mutex_;
  std::unordered_map<Context, std::vector<OpStatePtr>> cached_op_states_;
};

using CachedOpThreadSafePtr = std::shared_ptr<CachedOpThreadSafe>;

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_CACHED_OP_THREADSAFE_H_
