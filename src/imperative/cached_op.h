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

#ifndef MXNET_IMPERATIVE_CACHED_OP_H_
#define MXNET_IMPERATIVE_CACHED_OP_H_

#include <mxnet/imperative.h>
#include <vector>
#include <atomic>
#include <utility>
#include <string>
#include <unordered_map>

namespace mxnet {
/*! \brief CachedOp Parameters */
struct CachedOpConfig : public dmlc::Parameter<CachedOpConfig> {
  uint32_t inline_limit;
  uint32_t forward_bulk_size;
  uint32_t backward_bulk_size;
  bool static_alloc;
  bool static_shape;
  nnvm::Tuple<uint32_t> data_indices;
  nnvm::Tuple<uint32_t> param_indices;
  std::string subgraph;
  DMLC_DECLARE_PARAMETER(CachedOpConfig) {
    DMLC_DECLARE_FIELD(static_alloc)
    .set_default(false)
    .describe("Statically allocate memory to improve speed. "
              "Memory usage may increase.");
    DMLC_DECLARE_FIELD(static_shape)
    .set_default(false)
    .describe("Optimize for invariant input shapes between iterations. "
              "Must also set static_alloc to True. "
              "Change of input shapes is still allowed but slower.");
    DMLC_DECLARE_FIELD(inline_limit)
    .set_default(2)
    .describe("Maximum number of operators that can be inlined.");
    DMLC_DECLARE_FIELD(forward_bulk_size)
    .set_default(dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN", 15))
    .describe("Segment size of bulk execution during forward pass.");
    DMLC_DECLARE_FIELD(backward_bulk_size)
    .set_default(dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN", 15))
    .describe("Segment size of bulk execution during backward pass.");
    DMLC_DECLARE_FIELD(data_indices)
    .set_default(nnvm::Tuple<uint32_t>())
    .describe("Position of argument variables.");
    DMLC_DECLARE_FIELD(param_indices)
    .set_default(nnvm::Tuple<uint32_t>())
    .describe("Position of parameters.");
    DMLC_DECLARE_FIELD(subgraph)
    .set_default(std::string(""))
    .describe("JSON string of a subgraph.");
  }
};

class CachedOp {
 public:
  CachedOp(
      const nnvm::Symbol& sym,
      const std::vector<std::pair<std::string, std::string> >& flags);
  ~CachedOp();
  uint32_t num_inputs() const {
    return fwd_graph_.indexed_graph().input_nodes().size();
  }
  uint32_t num_outputs() const {
    return fwd_graph_.outputs.size();
  }
  uint32_t num_backward_inputs() const {
    return bwd_ograd_dep_.size() + bwd_in_dep_.size() + bwd_out_dep_.size();
  }
  uint32_t num_backward_outputs() const {
    auto &idx = fwd_graph_.indexed_graph();
    return idx.input_nodes().size() - idx.mutable_input_nodes().size();
  }
  std::vector<bool>& save_inputs() {
    return save_inputs_;
  }
  std::vector<bool>& save_outputs() {
    return save_outputs_;
  }
  const std::unordered_set<uint32_t>& mutable_input_nodes() const {
    return fwd_graph_.indexed_graph().mutable_input_nodes();
  }
  std::vector<nnvm::NodeEntry> Gradient(
      const nnvm::NodePtr& node,
      const std::vector<nnvm::NodeEntry>& ograds) const;
  OpStatePtr Forward(
      const std::shared_ptr<CachedOp>& op_ptr,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs);
  void Backward(
      const bool retain_graph,
      const OpStatePtr& state,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs);
  // backward storage type inference
  bool BackwardStorageType(
      const nnvm::NodeAttrs& attrs,
      const int dev_mask,
      DispatchMode* dispatch_mode,
      std::vector<int> *in_attrs,
      std::vector<int> *out_attrs);
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

 private:
  struct GraphInfo;
  struct DynamicRuntime;
  struct CachedOpState;

  OpStatePtr GetCachedOpState(const Context& ctx);
  bool SetForwardGraph(
      GraphInfo* info,
      const bool recording,
      const std::vector<NDArray*>& inputs);
  bool SetBackwardGraph(
      GraphInfo* info,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& inputs,
      bool detect_inplace_addto = false);
  OpStatePtr DynamicForward(
      const Context& default_ctx,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs);
  void DynamicBackward(
      const bool retain_graph,
      const OpStatePtr& op_state,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs);
  void StaticAllocMemory(
      const OpStatePtr& state_ptr,
      bool recording,
      bool keep_fwd);
  void StaticInitExec(
      const OpStatePtr& state_ptr,
      bool recording,
      bool keep_fwd);
  void StaticRunOps(
      const Context& default_ctx,
      const nnvm::Graph& g,
      const OpStatePtr& state_ptr,
      const std::vector<NDArray *> &state_arrays,
      size_t start_nid,
      size_t end_nid);
  OpStatePtr StaticForward(
      const Context& default_ctx,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs);
  void StaticBackward(
      const bool retain_graph,
      const OpStatePtr& state_ptr,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs);

  CachedOpConfig config_;
  nnvm::Graph fwd_graph_;
  nnvm::Graph grad_graph_;
  nnvm::Graph full_graph_;
  bool inlining_;
  std::vector<nnvm::NodeEntry> ograd_entries_;
  std::vector<uint32_t> bwd_in_dep_, bwd_out_dep_, bwd_ograd_dep_;
  std::unordered_map<uint32_t, uint32_t> fwd_input_to_grad_output_;
  std::vector<bool> save_inputs_, save_outputs_;
  std::vector<OpReqType> bwd_output_reqs_;

  std::mutex mutex_;
  std::unordered_map<Context, std::vector<OpStatePtr> > cached_op_states_;
};

using CachedOpPtr = std::shared_ptr<CachedOp>;

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_CACHED_OP_H_
