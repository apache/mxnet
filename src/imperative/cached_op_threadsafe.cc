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
#include "../executor/exec_pass.h"
#include "./cached_op_threadsafe.h"
#include "../profiler/profiler.h"
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"

namespace mxnet {
DMLC_REGISTER_PARAMETER(CachedOpThreadSafeConfig);

constexpr uint32_t kEidNotExist = std::numeric_limits<uint32_t>::max();

struct CachedOpThreadSafe::GraphInfo {
  nnvm::Graph fwd_graph;
};

struct CachedOpThreadSafe::DynamicRuntime {
  GraphInfo info;
  std::vector<OpStatePtr> op_states;
};

struct CachedOpThreadSafe::CachedOpThreadSafeState {
  CachedOpThreadSafeState(const Context &context_,
                          const nnvm::Graph &fwd_graph_) {
    context = context_;
    info.fwd_graph = fwd_graph_;

    size_t max_nodes = info.fwd_graph.indexed_graph().num_nodes();
    size_t max_entries = info.fwd_graph.indexed_graph().num_node_entries();
    info.fwd_graph.attrs["context"] =
      std::make_shared<dmlc::any>(std::vector<Context>(
                     info.fwd_graph.indexed_graph().num_nodes(), context));

    buff.resize(max_entries);
    arrays.resize(max_entries);
    array_reqs.resize(max_entries);
    dynamic_entries.resize(max_entries, false);
    op_states.resize(max_nodes);
    execs.resize(max_nodes);
    opr_segs.resize(max_nodes);
  }

  std::mutex mutex;
  Context context;
  GraphInfo info;
  bool fwd_alloc = false;
  bool fwd_exec_init = false;

  std::vector<NDArray> buff;
  std::vector<NDArray*> arrays;
  std::vector<NDArray*> arrays_with_in_out;
  std::vector<OpReqType> array_reqs;
  std::vector<std::shared_ptr<exec::OpExecutor> > execs;
  std::vector<imperative::EngineOprSeg> opr_segs;
  std::vector<OpStatePtr> op_states;

  std::vector<bool> dynamic_entries;
  std::multimap<size_t, NDArray> fwd_reuse_pool;
};

  OpStatePtr CachedOpThreadSafe::GetCachedOpThreadSafeState(
                                 const Context& ctx) {
    for (const auto& i : cached_op_states_[ctx]) {
      // only create one state per device when not using static memory
      if (!config_.static_alloc || i.unique()) {
        return i;
      }
    }
    auto state_ptr = OpStatePtr::Create<CachedOpThreadSafeState>(ctx, fwd_graph_);

    cached_op_states_[ctx].push_back(state_ptr);
    return state_ptr;
  }

  CachedOpThreadSafe::CachedOpThreadSafe(const nnvm::Symbol& sym,
                                         const std::vector<std::pair<std::string,
                                         std::string> >& flags) {
    using namespace nnvm;
    using namespace imperative;
    static const std::vector<const Op *> zero_ops{Op::Get("zeros_like"),
        Op::Get("_zeros")};
    static const auto _copy_op = Op::Get("_copy");
    config_.Init(flags);

    if (config_.static_shape) {
      CHECK(config_.static_alloc) << "static_alloc must be True when static_shape is True";
    }

    // construct forward graph
    {
      NodeEntryMap<size_t> dedup_out;
      for (const NodeEntry &nodeEntry : sym.outputs) {
        if (dedup_out.find(nodeEntry) != dedup_out.end()) {
          NodePtr copy_node = Node::Create();
          copy_node->attrs.op = _copy_op;
          copy_node->attrs.name = nodeEntry.node->attrs.name + "_copy" +
            std::to_string(dedup_out[nodeEntry]++);
          copy_node->inputs.emplace_back(nodeEntry);
          if (_copy_op->attr_parser != nullptr) {
            _copy_op->attr_parser(&(copy_node->attrs));
          }
          fwd_graph_.outputs.emplace_back(std::move(copy_node));
        } else {
          dedup_out.emplace(nodeEntry, 0);
          fwd_graph_.outputs.push_back(nodeEntry);
        }
      }

      const auto &idx = fwd_graph_.indexed_graph();
      CHECK_GE(idx.input_nodes().size(), 1)
        << "CachedOp requires at least 1 input";

      std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
      for (const auto &i : idx.input_nodes())
        ++ref_count[idx.entry_id(i, 0)];
      for (const auto &i : idx.outputs())
        ++ref_count[idx.entry_id(i)];
      for (size_t i = 0; i < idx.num_nodes(); ++i) {
        for (const auto &j : idx[i].inputs)
          ++ref_count[idx.entry_id(j)];
      }

      fwd_graph_.attrs["forward_ref_count"] =
        std::make_shared<dmlc::any>(std::move(ref_count));
    }

    // Set param indices
    {
      const auto& indexed_graph = fwd_graph_.indexed_graph();
      if (config_.data_indices.ndim() || config_.param_indices.ndim()) {
        CHECK_EQ(config_.data_indices.ndim() + config_.param_indices.ndim(),
                 indexed_graph.input_nodes().size());
      } else {
        std::vector<uint32_t> tmp;
        tmp.reserve(indexed_graph.input_nodes().size());
        for (size_t i = 0; i < indexed_graph.input_nodes().size(); ++i) {
          tmp.emplace_back(i);
        }
        config_.data_indices.assign(tmp.begin(), tmp.end());
      }
    }
  }

  bool CachedOpThreadSafe::SetForwardGraph(GraphInfo *info,
                                           const std::vector<NDArray *> &inputs) {
    using namespace nnvm;
    using namespace imperative;
    CHECK_EQ(inputs.size(), num_inputs());
    nnvm::Graph& g = info->fwd_graph;

    ShapeVector shape_inputs;
    DTypeVector dtype_inputs;
    StorageTypeVector storage_type_inputs;
    shape_inputs.reserve(inputs.size());
    dtype_inputs.reserve(inputs.size());
    storage_type_inputs.reserve(inputs.size());
    for (auto input : inputs) {
      shape_inputs.emplace_back(input->shape());
      dtype_inputs.emplace_back(input->dtype());
      storage_type_inputs.emplace_back(input->storage_type());
    }

    bool match = true;
    bool contain_dynamic_shape = false;
    match &= CheckAndInferShape(&g, std::move(shape_inputs), true,
                                {0, 0}, {0, 0}, &contain_dynamic_shape);
    match &= CheckAndInferType(&g, std::move(dtype_inputs), true);
    exec::DevMaskVector dev_mask(g.indexed_graph().num_nodes(), inputs[0]->ctx().dev_mask());
    match &= CheckAndInferStorageType(&g, std::move(dev_mask),
                                     std::move(storage_type_inputs), true);

    if (!match) {
      g.attrs.erase("forward_mem_plan");
    } else if (g.attrs.count("forward_mem_plan")) {
      return true;
    }

    const auto& idx = g.indexed_graph();

    StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
    const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
    CHECK_EQ(stypes.size(), storage.size());

    for (size_t i = 0; i < stypes.size(); i++) {
      if (stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
    }

    for (const auto i : idx.input_nodes()) {
      storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
    }

    for (size_t i = 0; i < idx.outputs().size(); ++i) {
      storage[idx.entry_id(idx.outputs()[i])] = exec::kExternalStorageID;
    }

    auto mem_plan = MXPlanMemory(&g, std::move(storage),
                                 g.GetAttr<std::vector<uint32_t>>("forward_ref_count"),
                                 "forward_storage_plan");
    g.attrs["forward_mem_plan"] =
      std::make_shared<dmlc::any>(std::move(mem_plan));

    return false;
  }

  void CachedOpThreadSafe::StaticAllocMemory(const OpStatePtr& state_ptr) {
    using namespace nnvm;
    using namespace imperative;

    auto& state = state_ptr.get_state<CachedOpThreadSafeState>();
    const auto& default_ctx = state.context;
    nnvm::Graph& g = state.info.fwd_graph;
    const auto& idx = g.indexed_graph();
    const auto& storage_plan = g.GetAttr<std::vector<int> >("forward_storage_plan");
    const auto& mem_plan = g.GetAttr<MemoryPlanVector>("forward_mem_plan");
    std::vector<int> addto_entry;
    if (g.attrs.count("addto_entry")) {
      addto_entry = g.GetAttr<std::vector<int>>("addto_entry");
    }
    size_t start_eid = 0;
    size_t end_eid = idx.num_node_entries();

    state.fwd_alloc = false;

    for (size_t i = start_eid; i < state.buff.size(); ++i) {
      state.buff[i] = NDArray();
      state.arrays[i] = &state.buff[i];
      state.array_reqs[i] = kNullOp;
      state.dynamic_entries[i] = false;
    }

    for (auto i : idx.input_nodes()) {
      auto eid = idx.entry_id(i, 0);
      if (eid >= start_eid)
        state.dynamic_entries[eid] = true;
    }

    for (auto i : idx.outputs()) {
      auto eid = idx.entry_id(i);
      if (eid >= start_eid) state.dynamic_entries[eid] = true;
    }

    for (size_t i = start_eid; i < end_eid; ++i) {
      if (addto_entry.size() && addto_entry[i]) {
        state.array_reqs[i] = kAddTo;
      } else if (storage_plan[i] >= 0) {
        state.array_reqs[i] = kWriteInplace;
      } else if (storage_plan[i] == -2) {
        state.array_reqs[i] = kNullOp;
      } else {
        state.array_reqs[i] = kWriteTo;
      }
    }

    auto& reuse_pool = state.fwd_reuse_pool;
    reuse_pool = imperative::AllocateMemory(
                 g, idx, default_ctx, start_eid, end_eid, mem_plan, state.arrays,
                 &state.array_reqs, std::move(reuse_pool));

    state.fwd_alloc = true;
  }

  void CachedOpThreadSafe::StaticInitExec(const OpStatePtr &state_ptr) {
    using namespace nnvm;
    using namespace imperative;

    auto &state = state_ptr.get_state<CachedOpThreadSafeState>();
    const auto &default_ctx = state.context;
    nnvm::Graph &g = state.info.fwd_graph;
    const auto &idx = g.indexed_graph();
    size_t start_nid = 0;
    size_t end_nid = idx.num_nodes();
    std::vector<int> skip_plus_node;
    if (g.attrs.count("skip_plus_node")) {
      skip_plus_node = g.GetAttr<std::vector<int> >("skip_plus_node");
    }

    state.fwd_exec_init = false;

    for (size_t i = start_nid; i < state.execs.size(); ++i) {
      state.execs[i].reset();
      state.opr_segs[i] = EngineOprSeg();
    }

    if (!config_.static_shape) {
      for (size_t i = start_nid; i < end_nid; ++i) {
        state.opr_segs[i].next_nid = i + 1;
        state.opr_segs[i].skip = skip_plus_node.size() && skip_plus_node[i];
      }
    } else {
      for (size_t i = start_nid; i < state.execs.size(); ++i) {
        exec::CreateOpExecs(g, &state.execs, &state.op_states, i);
      }
      exec::AttachOpResources(g, state.execs, start_nid, end_nid);

      for (size_t i = start_nid; i < end_nid; ++i) {
        bool skip = idx[i].source->is_variable();
        for (size_t j = 0; !skip && j < idx[i].inputs.size(); ++j) {
          skip = state.dynamic_entries[idx.entry_id(idx[i].inputs[j])];
        }
        for (size_t j = 0; !skip && j < idx[i].source->num_outputs(); ++j) {
          skip = state.dynamic_entries[idx.entry_id(i, j)];
        }
        if (skip)
          continue;
        SetupOpExec(g, i, state.execs[i], state.arrays, state.array_reqs);
      }

      CreateEngineOpSeg(idx, default_ctx, start_nid, end_nid, 0,
                        state.execs, skip_plus_node, &state.opr_segs);
    }
    state.fwd_exec_init = true;
  }

  void CachedOpThreadSafe::StaticRunOps(const Context &default_ctx,
                                        const nnvm::Graph &g,
                                        const OpStatePtr &state_ptr,
                                        const std::vector<NDArray *> &state_arrays,
                                        size_t start_nid, size_t end_nid) {
    static auto &createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");

    bool profiling =
      profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
    auto &state = state_ptr.get_state<CachedOpThreadSafeState>();
    const auto& idx = g.indexed_graph();
    const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
    const auto& op_execs = state.execs;

    std::vector<NDArray *> ndinputs, ndoutputs;
    mxnet::ShapeVector arg_shapes;
    nnvm::DTypeVector arg_dtypes;
    std::vector<OpReqType> req;

    for (size_t i = start_nid; config_.static_shape && i < end_nid; ++i) {
      if (op_execs[i]) op_execs[i]->op_ctx.is_train = false;
    }

    for (size_t i = start_nid; i < end_nid; i = state.opr_segs[i].next_nid) {
      const auto &opr_seg = state.opr_segs[i];
      if (opr_seg.skip)
        continue;
      if (opr_seg.opr != nullptr) {
        Engine::Get()->Push(opr_seg.opr.get(), default_ctx, 0, profiling);
      } else {
        const nnvm::IndexedGraph::Node &node = idx[i];
        if (node.source->is_variable())
          continue;
        auto num_outputs = node.source->num_outputs();
        ndinputs.clear();
        ndinputs.reserve(node.inputs.size());
        for (const auto &j : node.inputs) {
          ndinputs.emplace_back(state_arrays[idx.entry_id(j)]);
          CHECK(!ndinputs.back()->is_none());
        }
        ndoutputs.clear();
        ndoutputs.reserve(num_outputs);
        req.clear();
        req.reserve(num_outputs);
        for (size_t j = 0; j < num_outputs; ++j) {
          size_t eid = idx.entry_id(i, j);
          ndoutputs.emplace_back(state_arrays[eid]);
          req.push_back(state.array_reqs[eid]);
          CHECK(req.back() == kNullOp || !ndoutputs.back()->is_none());
        }
        const DispatchMode dispatch_mode = dispatch_modes[i];

        if (createop.count(node.source->op())) {
          arg_shapes.clear();
          arg_dtypes.clear();
          arg_shapes.reserve(ndinputs.size());
          arg_dtypes.reserve(ndinputs.size());
          for (auto &ndinput : ndinputs) {
            arg_shapes.emplace_back(ndinput->shape());
            arg_dtypes.emplace_back(ndinput->dtype());
          }
          if (!config_.static_shape) {
            state.op_states[i] = createop[node.source->op()](
                      node.source->attrs, default_ctx, arg_shapes, arg_dtypes);
          }
          Imperative::Get()->InvokeOp(default_ctx, node.source->attrs, ndinputs,
                                      ndoutputs, req, dispatch_mode,
                                      state.op_states[i]);
        } else {
          Imperative::Get()->InvokeOp(default_ctx, node.source->attrs, ndinputs,
                                      ndoutputs, req, dispatch_mode);
        }
      }
    }
  }

  OpStatePtr CachedOpThreadSafe::StaticForward(const Context &default_ctx,
                                               const std::vector<NDArray *> &inputs,
                                               const std::vector<NDArray *> &outputs) {
    using namespace nnvm;
    using namespace imperative;

    auto state_ptr = GetCachedOpThreadSafeState(default_ctx);
    auto &state = state_ptr.get_state<CachedOpThreadSafeState>();

    // Need to lock the mutex on the state, this allows
    // for multi context push of ops to dependency engine.
    // Required to lock for the whole function since static
    // alloc allocates memory, and executors once and reuses the alloced memory
    // and executors for multiple forward invokes of the same op.
    std::lock_guard<std::mutex> lock(state.mutex);

    bool match = SetForwardGraph(&state.info, inputs);

    nnvm::Graph &g = state.info.fwd_graph;
    const auto &idx = g.indexed_graph();

    if (!state.fwd_alloc || !match) {
      StaticAllocMemory(state_ptr);
    }

    state.arrays_with_in_out = state.arrays;
    auto &arrays = state.arrays_with_in_out;

    if (config_.static_shape) {
      for (auto i : config_.param_indices) {
        auto nid = idx.input_nodes()[i];
        if (!arrays[idx.entry_id(nid, 0)]->IsSame(*inputs[i])) {
          match = false;
          auto ptr = &state.buff[idx.entry_id(nid, 0)];
          CHECK_EQ(arrays[idx.entry_id(nid, 0)], ptr);
          *arrays[idx.entry_id(nid, 0)] = *inputs[i];
          state.dynamic_entries[idx.entry_id(nid, 0)] = false;
        }
      }
      for (auto i : config_.data_indices) {
        auto eid = idx.entry_id(idx.input_nodes()[i], 0);
        arrays[eid] = inputs[i];
      }
    } else {
      for (size_t i = 0; i < num_inputs(); ++i) {
        auto nid = idx.input_nodes()[i];
        arrays[idx.entry_id(nid, 0)] = inputs[i];
      }
    }

    if (!state.fwd_exec_init || !match) {
      StaticInitExec(state_ptr);
    }

    const auto &dtypes = g.GetAttr<DTypeVector>("dtype");
    const auto &shapes = g.GetAttr<mxnet::ShapeVector>("shape");
    const auto &stypes = g.GetAttr<StorageTypeVector>("storage_type");

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      // An input and output may share the same array.
      if (!arrays[eid]->is_none())
        *outputs[i] = arrays[eid]->Detach();
      arrays[eid] = outputs[i];
      if (!outputs[i]->is_none())
        continue;
      *outputs[i] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                            shapes[eid], default_ctx, true, dtypes[eid]);
    }

    StaticRunOps(default_ctx, g, state_ptr, arrays, 0, idx.num_nodes());

    return OpStatePtr();
  }

  OpStatePtr CachedOpThreadSafe::DynamicForward(const Context& default_ctx,
                                                const std::vector<NDArray*>& inputs,
                                                const std::vector<NDArray*>& outputs) {
    using namespace nnvm;
    using namespace imperative;

    {
      auto state_ptr = GetCachedOpThreadSafeState(default_ctx);
      auto op_state = OpStatePtr::Create<DynamicRuntime>();
      auto &runtime = op_state.get_state<DynamicRuntime>();
      {
        auto &state = state_ptr.get_state<CachedOpThreadSafeState>();
        // Need to lock the mutex on the state, this allows
        // for multi context push of ops to dependency engine.
        // SetForwardGraph runs infer passes on graphs as well
        // as the planmemory pass.
        std::lock_guard<std::mutex> lock(state.mutex);
        SetForwardGraph(&state.info, inputs);
        runtime.info.fwd_graph = state.info.fwd_graph;
      }
      nnvm::Graph &g = runtime.info.fwd_graph;
      const auto &idx = g.indexed_graph();
      size_t num_inputs = idx.input_nodes().size();
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
      for (size_t i = 0; i < num_inputs; ++i) {
        arrays[idx.entry_id(idx.input_nodes()[i], 0)] = inputs[i];
      }
      for (size_t i = 0; i < idx.outputs().size(); ++i) {
        auto eid = idx.entry_id(idx.outputs()[i]);
        if (!arrays[eid]->is_none())
          *outputs[i] = arrays[eid]->Detach();
        arrays[eid] = outputs[i];
      }
      // Allocate NDArrays
      std::vector<uint32_t> ref_count = g.GetAttr<std::vector<uint32_t>>(
                                                          "forward_ref_count");

      std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
      for (size_t i = 0; i < idx.num_node_entries(); ++i) {
        if (ref_count[i] == 0)
          array_reqs[i] = kNullOp;
      }
      const auto &dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
      const auto &mem_plan = g.GetAttr<MemoryPlanVector>("forward_mem_plan");
      AllocateMemory(g, idx, default_ctx, 0, idx.num_node_entries(), mem_plan,
                     arrays, &array_reqs);
      const auto &dtypes = g.GetAttr<DTypeVector>("dtype");
      const auto &shapes = g.GetAttr<mxnet::ShapeVector>("shape");
      const auto &stypes = g.GetAttr<StorageTypeVector>("storage_type");
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto eid = idx.entry_id(idx.outputs()[i]);
        arrays[eid] = outputs[i];
        if (!outputs[i]->is_none())
          continue;
        *outputs[i] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                              shapes[eid], default_ctx, true, dtypes[eid]);
      }
      // If CachedOp is running in the inline mode, it uses RunGraph to record
      // computation; otherwise, CachedOp records computation itself.
      // So if it's not the inline mode, we disable recording.
      RunGraph(false, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
               std::move(ref_count), &states, dispatch_modes, false);
      return op_state;
    }
  }

  OpStatePtr CachedOpThreadSafe::Forward(const std::shared_ptr<CachedOpThreadSafe>& op_ptr,
                                         const std::vector<NDArray*>& inputs,
                                         const std::vector<NDArray*>& outputs) {
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
    Context default_ctx = inputs[0]->ctx();
    const auto& idx = fwd_graph_.indexed_graph();
    for (size_t i = 0; i < inputs.size(); ++i) {
      CHECK_EQ(inputs[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name
        << " is on " << default_ctx << " while "
        << idx[idx.input_nodes()[i]].source->attrs.name
        << " is on " << inputs[i]->ctx();
    }

    OpStatePtr op_state;
    try {
      if (config_.static_alloc) {
        op_state = StaticForward(default_ctx, inputs, outputs);
      } else {
        op_state = DynamicForward(default_ctx, inputs, outputs);
      }
    } catch (const dmlc::Error& e) {
      throw e;
    }
    return op_state;
  }

struct CachedOpThreadSafeActualState {
  std::shared_ptr<CachedOpThreadSafe> op;
  OpStatePtr forward_state;

  explicit CachedOpThreadSafeActualState(std::shared_ptr<CachedOpThreadSafe> op) {
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
    s.forward_state = s.op->Forward(nullptr, in_ptrs, out_ptrs);
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
  CachedOpThreadSafe::~CachedOpThreadSafe() {}

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
