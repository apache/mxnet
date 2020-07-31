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

#include <unordered_set>
#include <iostream>
#include "./imperative_utils.h"
#include "./exec_pass.h"
#include "./cached_op_threadsafe.h"
#include "../profiler/profiler.h"
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"

namespace mxnet {

DMLC_REGISTER_PARAMETER(CachedOpThreadSafeConfig);

struct CachedOpThreadSafe::GraphInfo {
  nnvm::Graph fwd_graph;
};

struct CachedOpThreadSafe::DynamicRuntime {
  GraphInfo info;
  std::vector<OpStatePtr> op_states;
};

OpStatePtr CachedOpThreadSafe::GetCachedOpState(
    const Context& ctx) {

  for (const auto& i : cached_op_states_[ctx]) {
    // only create one state per device when not using static memory
    if (!config_.static_alloc || i.unique()) {
      return i;
    }
  }
  nnvm::Graph full_graph;
  auto state_ptr = OpStatePtr::Create<CachedOpState>(ctx, fwd_graph_, full_graph, false);

  cached_op_states_[ctx].push_back(state_ptr);
  return state_ptr;
}


CachedOpThreadSafe::CachedOpThreadSafe(const nnvm::Symbol& sym,
                                       const std::vector<std::pair<std::string,
                                       std::string> >& flags) : CachedOp(sym, flags) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op *> zero_ops{Op::Get("zeros_like"),
                                                Op::Get("_zeros")};
  config_.Init(flags);

  if (config_.static_shape) {
      CHECK(config_.static_alloc) << "static_alloc must be True when static_shape is True";
  }

  // construct forward graph
  CreateForwardGraph(sym.Copy(), &fwd_graph_);
  SetForwardRefCounts(&fwd_graph_);

  SetInputIndices(fwd_graph_, config_.param_indices,
                  &config_.data_indices);
}

/*
 * \brief Thread safe version of DynamicForward, with thread local buffer
 * used to store intermediate nodes in the graph
 */
OpStatePtr CachedOpThreadSafe::DynamicForward(const Context& default_ctx,
                                              const std::vector<NDArray*>& inputs,
                                              const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  auto state_ptr = GetCachedOpState(default_ctx);
  auto op_state = OpStatePtr::Create<DynamicRuntime>();
  auto &runtime = op_state.get_state<DynamicRuntime>();
  {
    auto &state = state_ptr.get_state<CachedOpState>();
    // Need to lock the mutex on the state, this allows
    // for multi context push of ops to dependency engine.
    // SetForwardGraph runs infer passes on graphs as well
    // as the planmemory pass.
    std::lock_guard<std::mutex> lock(state.mutex);
    // the below call runs the NNVM graph passes: type inference,
    // shape inference, storage type inference and if the graph
    // doesn't have dynamic shapes it also plans and allocates memory
    // for intermediate and final outputs in the graph
    SetForwardGraph(default_ctx, &state.info, false, inputs);
    runtime.info.fwd_graph = state.info.fwd_graph;
  }
  nnvm::Graph &g = runtime.info.fwd_graph;
  const auto &idx = g.indexed_graph();
  size_t max_nodes = runtime.info.fwd_graph.indexed_graph().num_nodes();
  runtime.op_states.resize(max_nodes);
  auto &states = runtime.op_states;

  // Allocate entries
  // This buff is thread local and used to store intermediate
  // nodes in the graph
  buff.resize(idx.num_node_entries());
  states.resize(idx.num_nodes());
  std::vector<NDArray *> arrays;
  arrays.reserve(buff.size());
  for (auto &buffered_array : buff) {
    arrays.push_back(&buffered_array);
  }
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  const auto &dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  std::vector<uint32_t> ref_count = g.GetAttr<std::vector<uint32_t>>(
      "forward_ref_count");
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  const MemoryPlanVector& mem_plan = g.GetAttr<MemoryPlanVector>("forward_mem_plan");
  // Collect input output pointers to ndarray into the arrays data structure
  std::vector<size_t> input_map(inputs.size());
  std::iota(input_map.begin(), input_map.end(), 0);
  CollectInputOutputNDRefs(g, inputs, input_map, outputs, &arrays);
  // The SetForwardGraph call in DynamicForward runs the memory planning phase
  // and allocates storage for intermediate and final outputs of the graph
  // We need to still create NDArrays (pointer data structure), based on this
  // allocated memory from memory planning phase. The CreateGraphNDs below does
  // that.
  CreateGraphNDs(g, default_ctx, mem_plan, &array_reqs, &arrays);
  // Invokes operators in the graph in a topologically sorted manner
  RunGraph(false, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
           std::move(ref_count), &states, dispatch_modes, false);
  return op_state;
}

OpStatePtr CachedOpThreadSafe::Forward(const std::shared_ptr<CachedOp>& op_ptr,
                                       const std::vector<NDArray*>& inputs,
                                       const std::vector<NDArray*>& outputs,
                                       const Context& default_ctx) {
  // Acquiring lock on the mutex in forward
  // Without this there are issues with static_forward,
  // specifically with static_shape=True and dynamic_forward.
  // Adding the lock here for safety,
  // The perf hit would be acceptable because this involves just pushing
  // ops to engine and not actual execution
  // We are putting this lock here because without this there is a hang
  // in the accept4 call in CUDA lib.
  // TODO(anirudh2290): Investigate this issue more as it also prevents parallel
  // push of ops for different contexts
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(inputs.size(), num_inputs());
  const auto& idx = fwd_graph_.indexed_graph();
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name
        << " is on " << default_ctx << " while "
        << idx[idx.input_nodes()[i]].source->attrs.name
        << " is on " << inputs[i]->ctx();
  }

  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.forward_bulk_size);
  OpStatePtr op_state;
  try {
    if (CheckDynamicShapeExists(default_ctx, inputs, true)) {
      LOG(FATAL) << "Dynamic shapes aren't supported with thread-safe cached op";
    }
    if (config_.static_alloc) {
      op_state = StaticForward(default_ctx, inputs, outputs);
    } else {
      op_state = DynamicForward(default_ctx, inputs, outputs);
    }
  } catch (const dmlc::Error& e) {
    Engine::Get()->set_bulk_size(prev_bulk_size);
    throw e;
  }
  Engine::Get()->set_bulk_size(prev_bulk_size);
  return op_state;
}

struct CachedOpThreadSafeActualState {
  std::shared_ptr<CachedOp> op;
  OpStatePtr forward_state;

  explicit CachedOpThreadSafeActualState(std::shared_ptr<CachedOp> op) {
    this->op = op;
  }
};
OpStatePtr CreateCachedOpThreadSafeState(const NodeAttrs& attrs,
                               Context ctx,
                               const mxnet::ShapeVector& in_shapes,
                               const std::vector<int>& in_types) {
  const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
  return OpStatePtr::Create<CachedOpThreadSafeActualState>(op);
}

void CachedOpThreadSafeForward(const OpStatePtr& state_ptr,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  CachedOpThreadSafeActualState &s = state_ptr.get_state<CachedOpThreadSafeActualState>();
  std::vector<NDArray> in_bufs = inputs;
  std::vector<NDArray> out_bufs = outputs;
  std::vector<NDArray *> in_ptrs(in_bufs.size());
  std::vector<NDArray *> out_ptrs(out_bufs.size());
  for (size_t i = 0; i < in_ptrs.size(); i++)
    in_ptrs[i] = &in_bufs[i];
  for (size_t i = 0; i < out_ptrs.size(); i++)
    out_ptrs[i] = &out_bufs[i];

  // Set is_recording correct for the imperative executor.
  CHECK(!ctx.need_grad) << "Only inference use case supported with thread safe cached op";
  CHECK(!ctx.is_train) << "Only inference use case supported with thread safe cached op";
  CHECK(inputs.size() > 0) << "thread safe cached op requires at least one input";
  Context default_ctx = inputs[0].ctx();
  s.forward_state = s.op->Forward(nullptr, in_ptrs, out_ptrs, default_ctx);
  // The arrays in out_ptrs may be changed by CachedOp.
  // If it is, we need to copy data back.
  for (size_t i = 0; i < out_bufs.size(); i++)
    if (!out_bufs[i].IsSame(outputs[i]))
      CopyFromTo(out_bufs[i], outputs[i]);
}

void CachedOpThreadSafeParamParser(nnvm::NodeAttrs* attrs) {
  CachedOpThreadSafeConfig param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
}
CachedOpThreadSafe::~CachedOpThreadSafe() = default;

NNVM_REGISTER_OP(_CachedOpThreadSafe)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr_parser(CachedOpThreadSafeParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->ListForwardInputNames();
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->ListForwardOutputNames();
  })
.set_attr<FCreateOpState>("FCreateOpState", CreateCachedOpThreadSafeState)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_shapes,
     mxnet::ShapeVector *out_shapes) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpShapeHelper(op->GetForwardSym(), in_shapes, out_shapes);
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_types,
     std::vector<int> *out_types) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpTypeHelper(op->GetForwardSym(), in_types, out_types);
  })
.set_attr<FInferStorageType>("FInferStorageType",
  [](const nnvm::NodeAttrs& attrs,
     const int dev_mask,
     DispatchMode* dispatch_mode,
     std::vector<int>* in_stypes,
     std::vector<int>* out_stypes) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpStorageTypeHelper(op->GetForwardSym(),
                                                  dev_mask, dispatch_mode,
                                                  in_stypes, out_stypes);
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CachedOpThreadSafeForward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CachedOpThreadSafeForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpMutableInputsHelper(op->GetForwardSym());
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpResourceRequestHelper(op->GetForwardSym());
  })
.set_attr<FExecType>("FExecType", op::DefaultSubgraphOpExecType)
.add_argument("data", "NDArray-or-Symbol[]", "input data list");

}  // namespace mxnet
