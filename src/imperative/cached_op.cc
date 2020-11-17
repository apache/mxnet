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
#include <memory>
#include <unordered_set>
#include <iostream>
#include "./imperative_utils.h"
#include "./cached_op.h"
#include "./exec_pass.h"
#include "../profiler/profiler.h"
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"


namespace mxnet {

DMLC_REGISTER_PARAMETER(CachedOpConfig);

constexpr uint32_t kEidNotExist = std::numeric_limits<uint32_t>::max();

nnvm::Symbol CachedOp::GetOptimizedSymbol() const {
  nnvm::Symbol ret;
  ret.outputs = std::vector<nnvm::NodeEntry>(full_graph_.outputs.begin(),
                                             full_graph_.outputs.begin() + num_outputs());
  return ret.Copy();
}

CachedOp::CachedOp(
    const nnvm::Symbol& sym,
    const std::vector<std::pair<std::string, std::string> >& flags) : sym_(sym), flags_(flags) {
  config_.Init(flags);
  this->dynamic_shape_checked_ = false;

  if (config_.static_shape) {
    CHECK(config_.static_alloc) << "static_alloc must be True when static_shape is True";
  }

  auto grad_graph = nnvm::Graph();
  std::unordered_map<uint32_t, uint32_t> fwd_input_to_grad_output;
  CreateFullGraph(sym.Copy(), &fwd_graph_, &grad_graph, &full_graph_,
                  &ograd_entries_, &fwd_input_to_grad_output);

  {
    const auto& idx = fwd_graph_.indexed_graph();
    bwd_output_reqs_ = std::vector<OpReqType>(grad_graph.outputs.size(), kWriteTo);
    inlining_ = !config_.static_alloc &&
        (idx.num_nodes() - idx.input_nodes().size()) <= config_.inline_limit;
  }

  SetInputIndices(fwd_graph_, config_.param_indices, &config_.data_indices);

  // Set the backward dependency vectors
  {
    const auto& idx = full_graph_.indexed_graph();
    size_t num_forward_inputs = num_inputs();
    size_t num_forward_outputs = num_outputs();
    for (uint32_t i = 0; i < ograd_entries_.size(); ++i) {
      if (!idx.exist(ograd_entries_[i].node.get())) continue;
      bwd_ograd_dep_.push_back(i);
    }
    save_inputs_.resize(num_forward_inputs, false);
    for (uint32_t i = 0; i < num_forward_inputs; ++i) {
      save_inputs_[i] = true;
      bwd_in_dep_.push_back(i);
    }
    save_outputs_.resize(idx.outputs().size(), false);
    for (uint32_t i = 0; i < num_forward_outputs; ++i) {
      save_outputs_[i] = true;
      bwd_out_dep_.push_back(i);
    }
  }

  SetRefCounts(&fwd_graph_, full_graph_);
}

CachedOp::~CachedOp() = default;

std::vector<nnvm::NodeEntry> CachedOp::Gradient(
    const nnvm::ObjectPtr& node,
    const std::vector<nnvm::NodeEntry>& ograds) const {
  using namespace nnvm;
  static const auto _backward_CachedOp = Op::Get("_backward_CachedOp");
  static const auto _NoGrad = Op::Get("_NoGradient");

  auto p = Node::Create();
  p->attrs.op = _backward_CachedOp;
  p->attrs.name = node->attrs.name + "_backward";
  p->attrs.parsed = node->attrs.parsed;
  p->control_deps.push_back(node);
  p->inputs.reserve(bwd_ograd_dep_.size() + bwd_in_dep_.size() + bwd_out_dep_.size());
  for (auto i : bwd_ograd_dep_)
    p->inputs.push_back(ograds[i]);
  for (auto i : bwd_in_dep_)
    p->inputs.push_back(node->inputs[i]);
  for (auto i : bwd_out_dep_)
    p->inputs.emplace_back(node, i, 0);
  std::vector<NodeEntry> ret;
  ret.reserve(num_inputs());
  const auto& auxs = mutable_input_nodes();
  if (auxs.size()) {
    auto nop = Node::Create();
    nop->attrs.op = _NoGrad;
    nop->attrs.name = "NoGradient";
    uint32_t k = 0;
    for (const auto& i : fwd_graph_.indexed_graph().input_nodes()) {
      if (auxs.count(i)) {
        ret.emplace_back(nop);
      } else {
        ret.emplace_back(p, k++, 0);
      }
    }
  } else {
    for (uint32_t i = 0; i < num_inputs(); ++i)
        ret.emplace_back(p, i, 0);
  }
  return ret;
}

bool CachedOp::CheckDynamicShapeExists(const Context& default_ctx,
                                       const std::vector<NDArray*>& inputs,
                                       bool erase_result) {
  using namespace nnvm;
  using namespace imperative;
  if (this->dynamic_shape_checked_) {
    return config_.is_dynamic;
  } else {
    this->dynamic_shape_checked_ = true;
  }
  CHECK_EQ(inputs.size(), num_inputs());

  auto state_ptr = GetCachedOpState(default_ctx);
  auto& state = state_ptr.get_state<CachedOpState>();

  nnvm::Graph& g = state.info.fwd_graph;
  ShapeVector shape_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    shape_inputs[i] = inputs[state.info.input_map[i]]->shape();
  }
  // We leverage the shape inference pass to detect whether dynamic shape exists.
  // If so, the pass will fail with `contain_dynamic_shape = true`,
  // This method is only called once, so the overhead is negligible.
  bool contain_dynamic_shape = false;
  CheckAndInferShape(&g, std::move(shape_inputs), true,
                     {0, 0}, {0, 0},
                     &contain_dynamic_shape);
  if (!config_.static_shape && erase_result) {
    g.attrs.erase("shape");
    g.attrs.erase("shape_inputs");
  }
  return contain_dynamic_shape;
}

bool CachedOp::SetForwardGraph(
    const Context& default_ctx,
    GraphInfo* info,
    const bool recording,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  CHECK_EQ(inputs.size(), num_inputs());
  nnvm::Graph& g = info->fwd_graph;

  ShapeVector shape_inputs(inputs.size());
  DTypeVector dtype_inputs(inputs.size());
  StorageTypeVector storage_type_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    shape_inputs[i] = inputs[info->input_map[i]]->shape();
    dtype_inputs[i] = inputs[info->input_map[i]]->dtype();
    storage_type_inputs[i] = inputs[info->input_map[i]]->storage_type();
  }

  bool match = true;
  bool contain_dynamic_shape = false;
  match &= CheckAndInferShape(&g, std::move(shape_inputs), true,
                              {0, 0}, {0, 0}, &contain_dynamic_shape);
  match &= CheckAndInferType(&g, std::move(dtype_inputs), true);
  exec::DevMaskVector dev_mask(g.indexed_graph().num_nodes(), default_ctx.dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask),
                                    std::move(storage_type_inputs), true);

  // When dynmaic shape exists, it is not feasible to plan memory ahead of time
  if (contain_dynamic_shape) {
    g.attrs.erase(AddPrefix(FORWARD, MEM_PLAN));
    g.attrs.erase(AddPrefix(FULL, MEM_PLAN));
    return false;
  }
  const std::string& prefix = recording ? FULL : FORWARD;
  if (!match) {
    g.attrs.erase(AddPrefix(FORWARD, MEM_PLAN));
    g.attrs.erase(AddPrefix(FULL, MEM_PLAN));
  } else if (g.attrs.count(AddPrefix(prefix, MEM_PLAN))) {
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

  auto mem_plan = MXPlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >(AddPrefix(prefix, REF_COUNT)),
      AddPrefix(prefix, STORAGE_PLAN));
  g.attrs[AddPrefix(prefix, MEM_PLAN)] =
      std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}

// Utility function to set backward input eids
void SetBackwardInputEid(const std::vector<uint32_t>& bwd_in_dep,
                         const std::vector<uint32_t>& bwd_out_dep,
                         const std::vector<uint32_t>& bwd_ograd_dep,
                         const std::vector<nnvm::NodeEntry>& ograd_entries,
                         const nnvm::IndexedGraph& idx,
                         std::vector<uint32_t> *bwd_input_eid) {
  for (const auto& i : bwd_ograd_dep) {
    auto ograd = ograd_entries[i];
    if (idx.exist(ograd.node.get())) {
      bwd_input_eid->push_back(idx.entry_id(ograd));
    } else {
      bwd_input_eid->push_back(kEidNotExist);
    }
  }
  for (const auto& i : bwd_in_dep) {
    auto eid = idx.entry_id(idx.input_nodes()[i], 0);
    bwd_input_eid->push_back(eid);
  }
  for (const auto& i : bwd_out_dep) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    bwd_input_eid->push_back(eid);
  }
}

bool CachedOp::SetBackwardGraph(
    GraphInfo* info,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& inputs,
    bool detect_inplace_addto) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  Context default_ctx = inputs[0]->ctx();
  nnvm::Graph& g = info->full_graph;

  if (info->bwd_output_reqs != reqs) {
    info->bwd_output_reqs = reqs;
    info->bwd_input_eid.clear();
    g = nnvm::Graph();
    g.outputs = info->fwd_graph.outputs;
    for (size_t i = 0; i < info->grad_graph.outputs.size(); ++i) {
      if (info->bwd_output_reqs[i] == kNullOp) continue;
      g.outputs.emplace_back(info->grad_graph.outputs[i]);
    }
    g.attrs["context"] = std::make_shared<dmlc::any>(
        std::vector<Context>(g.indexed_graph().num_nodes(), default_ctx));
  }

  const auto& idx = g.indexed_graph();

  if (info->bwd_input_eid.size() != inputs.size()) {
    info->bwd_input_eid.clear();
    SetBackwardInputEid(bwd_in_dep_, bwd_out_dep_, bwd_ograd_dep_,
                        info->ograd_entries, idx, &info->bwd_input_eid);
    CHECK_EQ(inputs.size(), info->bwd_input_eid.size());
  }

  size_t num_forward_nodes = info->fwd_graph.indexed_graph().num_nodes();
  size_t num_forward_entries = info->fwd_graph.indexed_graph().num_node_entries();

  if (!g.attrs.count(AddPrefix(BACKWARD, REF_COUNT))) {
    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (info->bwd_input_eid[i] != kEidNotExist) {
        ++ref_count[info->bwd_input_eid[i]];
      }
    }
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    g.attrs[AddPrefix(BACKWARD, REF_COUNT)] = std::make_shared<dmlc::any>(std::move(ref_count));
  }

  auto shapes = info->fwd_graph.GetAttr<mxnet::ShapeVector>("shape");
  shapes.resize(idx.num_node_entries(), mxnet::TShape());
  auto dtypes = info->fwd_graph.GetAttr<DTypeVector>("dtype");
  dtypes.resize(idx.num_node_entries(), -1);
  auto stypes = info->fwd_graph.GetAttr<StorageTypeVector>("storage_type");
  stypes.resize(idx.num_node_entries(), -1);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (info->bwd_input_eid[i] == kEidNotExist) {
      continue;
    }
    size_t oi = BwdOriginalInput(info->input_map, i);
    shapes[info->bwd_input_eid[i]] = inputs[oi]->shape();
    dtypes[info->bwd_input_eid[i]] = inputs[oi]->dtype();
    stypes[info->bwd_input_eid[i]] = inputs[oi]->storage_type();
  }

  std::pair<uint32_t, uint32_t> node_range, entry_range;
  node_range = {num_forward_nodes, idx.num_nodes()};
  entry_range = {num_forward_entries, idx.num_node_entries()};

  bool match = true;
  match &= CheckAndInferShape(&g, std::move(shapes), false,
                              node_range, entry_range);
  match &= CheckAndInferType(&g, std::move(dtypes), false,
                             node_range, entry_range);
  exec::DevMaskVector dev_mask(idx.num_nodes(), default_ctx.dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask), std::move(stypes),
                                    false, node_range, entry_range);

  if (!match) {
    g.attrs.erase(AddPrefix(BACKWARD, MEM_PLAN));
  } else if (g.attrs.count(AddPrefix(BACKWARD, MEM_PLAN))) {
    return true;
  }

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  const auto& bwd_stypes = g.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = 0; i < bwd_stypes.size(); i++) {
    if (bwd_stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
  }
  for (size_t i = 0; i < num_forward_entries; ++i) storage[i] = exec::kExternalStorageID;
  for (const auto i : idx.input_nodes()) storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  for (const auto i : idx.outputs()) storage[idx.entry_id(i)] = exec::kExternalStorageID;

  auto mem_plan = MXPlanMemory(
      &g, std::move(storage),
      g.GetAttr<std::vector<uint32_t> >(AddPrefix(BACKWARD, REF_COUNT)),
      AddPrefix(BACKWARD, STORAGE_PLAN),
      {num_forward_nodes, idx.num_nodes()},
      {num_forward_entries, idx.num_node_entries()},
      detect_inplace_addto);
  g.attrs[AddPrefix(BACKWARD, MEM_PLAN)] = std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}

OpStatePtr CachedOp::GetCachedOpState(
    const Context& ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& i : cached_op_states_[ctx]) {
    // only create one state per device when not using static memory
    if (!config_.static_alloc || i.unique()) {
      return i;
    }
  }
  auto state_ptr = OpStatePtr::Create<CachedOpState>(ctx, fwd_graph_, full_graph_,
                                                     inlining_);

  cached_op_states_[ctx].push_back(state_ptr);
  return state_ptr;
}

void CachedOp::StaticAllocMemory(
    const OpStatePtr& state_ptr,
    bool recording,
    bool keep_fwd) {
  using namespace nnvm;
  using namespace imperative;

  auto& state = state_ptr.get_state<CachedOpState>();
  const auto& default_ctx = state.context;
  nnvm::Graph& g = keep_fwd ? state.info.full_graph : state.info.fwd_graph;
  const auto& idx = g.indexed_graph();
  const std::string& graph_type = keep_fwd ? BACKWARD : (recording ? FULL : FORWARD);
  const auto& storage_plan_attr = AddPrefix(graph_type, STORAGE_PLAN);
  const auto& storage_plan = g.GetAttr<std::vector<int> >(storage_plan_attr);
  const auto& mem_plan = g.GetAttr<MemoryPlanVector>(AddPrefix(graph_type, MEM_PLAN));
  std::vector<int> addto_entry;
  if (g.attrs.count("addto_entry")) {
    addto_entry = g.GetAttr<std::vector<int> >("addto_entry");
  }
  size_t start_eid =
      keep_fwd ? state.info.fwd_graph.indexed_graph().num_node_entries() : 0;
  size_t end_eid = idx.num_node_entries();

  if (!keep_fwd) state.fwd_alloc = false;
  state.bwd_alloc = false;
  for (size_t i = start_eid; i < state.buff.size(); ++i) {
    state.buff[i] = NDArray();
    state.arrays[i] = &state.buff[i];
    state.array_reqs[i] = kNullOp;
    state.dynamic_entries[i] = false;
  }

  for (auto i : idx.input_nodes()) {
    auto eid = idx.entry_id(i, 0);
    if (eid >= start_eid) state.dynamic_entries[eid] = true;
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
      // -2 indicate that the entry is never referenced.
      state.array_reqs[i] = kNullOp;
    } else {
      state.array_reqs[i] = kWriteTo;
    }
  }

  auto& reuse_pool = keep_fwd ? state.bwd_reuse_pool : state.fwd_reuse_pool;
  reuse_pool = imperative::AllocateMemory(
      g, idx, default_ctx, start_eid, end_eid, mem_plan,
      state.arrays, &state.array_reqs, std::move(reuse_pool));

  state.recording = recording;
  if (keep_fwd) {
    state.bwd_alloc = true;
  } else {
    state.fwd_alloc = true;
  }
}

void CachedOp::StaticInitExec(
    const OpStatePtr& state_ptr,
    bool recording,
    bool keep_fwd) {
  using namespace nnvm;
  using namespace imperative;

  auto& state = state_ptr.get_state<CachedOpState>();
  const auto& default_ctx = state.context;
  nnvm::Graph& g = keep_fwd ? state.info.full_graph : state.info.fwd_graph;
  const auto& idx = g.indexed_graph();
  std::vector<int> skip_plus_node;
  if (g.attrs.count("skip_plus_node")) {
    skip_plus_node = g.GetAttr<std::vector<int> >("skip_plus_node");
  }
  size_t start_nid =
      keep_fwd ? state.info.fwd_graph.indexed_graph().num_nodes() : 0;
  size_t end_nid = idx.num_nodes();

  if (!keep_fwd) state.fwd_exec_init = false;
  state.bwd_exec_init = false;

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
    for (size_t i = start_nid; i < end_nid; ++i) {
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
      if (skip) continue;
      SetupOpExec(g, i, state.execs[i], state.arrays, state.array_reqs);
    }

    // Init bulk_size for Inference mode with bulking enabled (= entire forward graph).
    size_t bulk_size = idx.num_nodes();
    if (recording || keep_fwd) {
      // Training mode
      if (!Imperative::PreferBulkExecTrain())
        bulk_size = 0;
      else
        bulk_size = keep_fwd ? config_.backward_bulk_size : config_.forward_bulk_size;
    } else {
      // Inference mode
      if (!Imperative::PreferBulkExecInference())
        bulk_size = 0;
    }

    CreateEngineOpSeg(idx, default_ctx, start_nid, end_nid, bulk_size,
                      state.execs, skip_plus_node, &state.opr_segs);
  }

  if (keep_fwd) {
    state.bwd_exec_init = true;
  } else {
    state.fwd_exec_init = true;
  }
}

void CachedOp::StaticRunOps(
    const Context& default_ctx,
    const nnvm::Graph& g,
    const OpStatePtr& state_ptr,
    const std::vector<NDArray *> &state_arrays,
    size_t start_nid,
    size_t end_nid) {
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");

  bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
  bool is_training = Imperative::Get()->is_training();
  auto& state = state_ptr.get_state<CachedOpState>();
  const auto& idx = g.indexed_graph();
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  const auto& op_execs = state.execs;

  std::vector<NDArray*> ndinputs, ndoutputs;
  mxnet::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_dtypes;
  std::vector<OpReqType> req;

  for (size_t i = start_nid; config_.static_shape && i < end_nid; ++i) {
    if (op_execs[i]) op_execs[i]->op_ctx.is_train = is_training;
  }

  for (size_t i = start_nid; i < end_nid; i = state.opr_segs[i].next_nid) {
    const auto& opr_seg = state.opr_segs[i];
    if (opr_seg.skip) continue;
    if (opr_seg.opr != nullptr) {
      Engine::Get()->Push(opr_seg.opr.get(), default_ctx, 0, profiling);
    } else {
      const nnvm::IndexedGraph::Node& node = idx[i];
      if (node.source->is_variable()) continue;
      auto num_outputs = node.source->num_outputs();
      ndinputs.clear();
      ndinputs.reserve(node.inputs.size());
      for (const auto& j : node.inputs) {
        ndinputs.emplace_back(state_arrays[idx.entry_id(j)]);
        CHECK(!ndinputs.back()->is_none());
      }
      if (monitor_callback_ && monitor_all_) {
          mxnet::common::ExecuteMonInputCallback(idx, state_arrays, i, monitor_callback_);
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
        for (auto& ndinput : ndinputs) {
          arg_shapes.emplace_back(ndinput->shape());
          arg_dtypes.emplace_back(ndinput->dtype());
        }
        if (!config_.static_shape) {
          state.op_states[i] =
              createop[node.source->op()](node.source->attrs, default_ctx, arg_shapes, arg_dtypes);
        }
        Imperative::Get()->InvokeOp(
            default_ctx, node.source->attrs, ndinputs, ndoutputs, req,
            dispatch_mode, state.op_states[i]);
      } else if (is_layer_backward.get(node.source->op(), false)) {
        nnvm::Node* fwd_node = node.source->control_deps[0].get();
        auto fwd_node_id = idx.node_id(fwd_node);
        Imperative::Get()->InvokeOp(
            default_ctx, node.source->attrs, ndinputs, ndoutputs,
            req, dispatch_mode, state.op_states[fwd_node_id]);
      } else {
        Imperative::Get()->InvokeOp(
            default_ctx, node.source->attrs, ndinputs, ndoutputs, req,
            dispatch_mode);
      }
      if (monitor_callback_) {
          mxnet::common::ExecuteMonOutputCallback(idx, state_arrays, i, monitor_callback_);
      }
    }
  }
}

#define INIT_DETACHED(x, y)   if (!y->is_none()) x->InitDetached(y)

static void PrepareOutputs(const nnvm::Graph& g, const Context& default_ctx,
                           const std::vector<NDArray*> &outputs,
                           std::vector<NDArray*> *pArrays, bool detach) {
  using namespace nnvm;
  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");

  const auto& idx = g.indexed_graph();
  auto &arrays = *pArrays;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto eid = idx.entry_id(idx.outputs()[i]);
    // An input and an output may share the same array.
    if (detach)
      INIT_DETACHED(outputs[i], arrays[eid]);

    arrays[eid] = outputs[i];
    if (arrays[eid]->is_none())
      arrays[eid]->ReInit(static_cast<NDArrayStorageType>(stypes[eid]),
                          shapes[eid], default_ctx, dtypes[eid]);
    const nnvm::NodeAttrs& attrs = idx[idx.outputs()[i].node_id].source->attrs;
    outputs[i]->AssignStorageInfo(common::NodeAttrsGetProfilerScope(attrs),
                                  attrs.name);
  }
}

OpStatePtr CachedOp::StaticForward(
    const Context& default_ctx,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  bool recording = Imperative::Get()->is_recording();
  auto state_ptr = GetCachedOpState(default_ctx);
  auto& state = state_ptr.get_state<CachedOpState>();

  // Need to lock the mutex on the state, this allows
  // for multi context push of ops to dependency engine.
  // Required to lock for the whole function since static
  // alloc allocates memory, and executors once and reuses the alloced memory
  // and executors for multiple forward invokes of the same op.
  std::lock_guard<std::mutex> lock(state.mutex);

  bool match = SetForwardGraph(default_ctx, &state.info, recording, inputs);
  match = match && state.recording == recording;

  nnvm::Graph& g = state.info.fwd_graph;
  const auto& idx = g.indexed_graph();
  if (!state.fwd_alloc || !match)  {
    StaticAllocMemory(state_ptr, recording, false);
  }

  // We are going to add input and output arrays to the array list.
  // The input and output arrays should only be valid for this run,
  // so we shouldn't modify the state's array list.
  state.arrays_with_in_out = state.arrays;
  auto& arrays = state.arrays_with_in_out;
  if (config_.static_shape) {
    for (auto i : config_.param_indices) {
      auto nid = idx.input_nodes()[i];
      if (!arrays[idx.entry_id(nid, 0)]->IsSame(*inputs[state.info.input_map[i]])) {
        match = false;
        auto ptr = &state.buff[idx.entry_id(nid, 0)];
        CHECK_EQ(arrays[idx.entry_id(nid, 0)], ptr);
        *arrays[idx.entry_id(nid, 0)] = *inputs[state.info.input_map[i]];
        state.dynamic_entries[idx.entry_id(nid, 0)] = false;
      }
    }
    for (auto i : config_.data_indices) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      arrays[eid] = inputs[state.info.input_map[i]];
    }
  } else {
    for (size_t i = 0; i < num_inputs(); ++i) {
      auto nid = idx.input_nodes()[i];
      arrays[idx.entry_id(nid, 0)] = inputs[state.info.input_map[i]];
    }
  }

  if (!state.fwd_exec_init || !match) {
    StaticInitExec(state_ptr, recording, false);
  }

  PrepareOutputs(g, default_ctx, outputs, &arrays, true);
  StaticRunOps(default_ctx, g, state_ptr, arrays, 0, idx.num_nodes());

  return recording ? state_ptr : OpStatePtr();
}


OpStatePtr CachedOp::DynamicForward(
    const Context& default_ctx,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    bool use_naive_run) {
  using namespace nnvm;
  using namespace imperative;

  // Initialize
  bool recording = Imperative::Get()->is_recording();
  auto op_state = OpStatePtr::Create<DynamicRuntime>();
  auto& runtime = op_state.get_state<DynamicRuntime>();
  {
    auto state_ptr = GetCachedOpState(default_ctx);
    auto& state = state_ptr.get_state<CachedOpState>();
    std::lock_guard<std::mutex> lock(state.mutex);
    SetForwardGraph(default_ctx, &state.info, recording, inputs);
    runtime.info.fwd_graph = state.info.fwd_graph;
    runtime.info.input_map = state.info.input_map;
  }
  nnvm::Graph& g = runtime.info.fwd_graph;
  const auto& idx = g.indexed_graph();
  auto& buff = runtime.buff;
  auto& states = runtime.op_states;

  // Allocate entries
  buff.resize(idx.num_node_entries());
  states.resize(idx.num_nodes());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (auto& buffered_array : buff) {
    arrays.push_back(&buffered_array);
  }
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  const std::string& graph_type = recording ? FULL : FORWARD;
  std::vector<uint32_t> ref_count =
    g.GetAttr<std::vector<uint32_t> >(AddPrefix(graph_type, REF_COUNT));
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }
  CollectInputOutputNDRefs(g, inputs, runtime.info.input_map, outputs, &arrays);

  if (!use_naive_run) {
    const auto& mem_plan = g.GetAttr<MemoryPlanVector >(AddPrefix(graph_type, MEM_PLAN));
    CreateGraphNDs(g, default_ctx, mem_plan, &array_reqs, &arrays);
    // If CachedOp is running in the inline mode, it uses RunGraph to record
    // computation; otherwise, CachedOp records computation itself.
    // So if it's not the inline mode, we disable recording.
    RunGraph(false, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
            std::move(ref_count), &states, dispatch_modes,
            recording && inlining_, nullptr, monitor_callback_, monitor_all_);
  } else {
    mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
    NaiveRunGraph(false, default_ctx, idx, arrays, 0, idx.num_nodes(),
                  std::move(array_reqs), std::move(ref_count), &states,
                  dispatch_modes, recording && inlining_, &shapes, monitor_callback_, monitor_all_);
    {
      auto state_ptr = GetCachedOpState(default_ctx);
      auto& state = state_ptr.get_state<CachedOpState>();
      auto copied_shape = shapes;
      std::lock_guard<std::mutex> lock(state.mutex);
      state.info.fwd_graph.attrs["shape"] = std::make_shared<dmlc::any>(std::move(copied_shape));
    }
    g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  }
  return op_state;
}

OpStatePtr CachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const Context& default_ctx) {
  static const auto cached_op = nnvm::Op::Get("_CachedOp");

  CHECK_EQ(inputs.size(), num_inputs());
  // Assign the storage information for the input arguments. Similar to the
  // implementation in `graph_executor.cc`, we use `mutable_input_nodes()` to
  // distinguish between weight parameters and auxiliary states.
  const auto& fwd_idx = fwd_graph_.indexed_graph();
  const auto& mutable_input_nodes = fwd_idx.mutable_input_nodes();
  for (size_t i = 0; i < fwd_idx.input_nodes().size(); ++i) {
    const uint32_t nid = fwd_idx.input_nodes().at(i);
    const nnvm::NodeAttrs& attrs = fwd_idx[nid].source->attrs;
    const std::string& arg_name = attrs.name;
    const std::string profiler_scope = common::NodeAttrsGetProfilerScope(attrs);
    if (mutable_input_nodes.count(nid)) {
      inputs[i]->AssignStorageInfo(profiler_scope + "aux_state:", arg_name);
    } else {
      inputs[i]->AssignStorageInfo(profiler_scope + "in_arg:", arg_name);
    }
  }

  {
    auto state_ptr = GetCachedOpState(default_ctx);
    auto& state = state_ptr.get_state<CachedOpState>();

    const auto& idx = state.info.fwd_graph.indexed_graph();
    for (size_t i = 0; i < inputs.size(); ++i) {
      CHECK_EQ(inputs[i]->ctx(), default_ctx)
          << "CachedOp requires all inputs to live on the same context. But "
          << idx[idx.input_nodes()[0]].source->attrs.name
          << " is on " << default_ctx << " while "
          << idx[idx.input_nodes()[i]].source->attrs.name
          << " is on " << inputs[i]->ctx();
    }
  }

  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.forward_bulk_size);

  OpStatePtr op_state;
  try {
    if (config_.is_dynamic || CheckDynamicShapeExists(default_ctx, inputs, true)) {
      config_.is_dynamic = true;
      config_.static_alloc = false;
      op_state = DynamicForward(default_ctx, inputs, outputs, true);
    } else if (config_.static_alloc) {
      op_state = StaticForward(default_ctx, inputs, outputs);
    } else {
      op_state = DynamicForward(default_ctx, inputs, outputs, false);
    }
  } catch (const dmlc::Error& e) {
    Engine::Get()->set_bulk_size(prev_bulk_size);
    throw e;
  }

  Engine::Get()->set_bulk_size(prev_bulk_size);

  if (Imperative::Get()->is_recording() && !inlining_) {
    nnvm::NodeAttrs attrs;
    attrs.op = cached_op;
    attrs.name = "_cachedop";
    attrs.parsed = op_ptr;
    Imperative::Get()->RecordOp(
        std::move(attrs), inputs, outputs, op_state,
        &save_inputs(), &save_outputs());
  }
  return op_state;
}

void CachedOp::DynamicBackward(
    const bool retain_graph,
    const OpStatePtr& op_state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  // Initialize
  Context default_ctx = outputs[0]->ctx();
  auto& runtime = op_state.get_state<DynamicRuntime>();
  {
    auto state_ptr = GetCachedOpState(default_ctx);
    auto& state = state_ptr.get_state<CachedOpState>();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.info.fwd_graph = runtime.info.fwd_graph;
    state.info.input_map = runtime.info.input_map;
    SetBackwardGraph(&state.info, reqs, inputs);
    runtime.info.full_graph = state.info.full_graph;
    runtime.info.bwd_input_eid = state.info.bwd_input_eid;
  }
  nnvm::Graph& g = runtime.info.full_graph;
  const auto& idx = g.indexed_graph();
  auto& buff = runtime.buff;
  auto& states = runtime.op_states;

  size_t num_forward_outputs = runtime.info.fwd_graph.outputs.size();
  size_t num_forward_nodes = runtime.info.fwd_graph.indexed_graph().num_nodes();
  size_t num_forward_entries = runtime.info.fwd_graph.indexed_graph().num_node_entries();
  buff.resize(idx.num_node_entries());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (auto& buffered_array : buff) {
    arrays.push_back(&buffered_array);
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (runtime.info.bwd_input_eid[i] == kEidNotExist) {
      continue;
    }
    arrays[runtime.info.bwd_input_eid[i]] = inputs[BwdOriginalInput(runtime.info.input_map, i)];
  }
  for (size_t i = 0, j = num_forward_outputs; i < reqs.size(); ++i) {
    if (reqs[i] == kNullOp) continue;
    const auto eid = idx.entry_id(idx.outputs()[j++]);
    // An input and an output may share the same array.
    INIT_DETACHED(outputs[i], arrays[eid]);
    arrays[eid] = outputs[i];
  }

  // Allocate NDArrays
  auto ref_count = g.GetAttr<std::vector<uint32_t> >(AddPrefix(BACKWARD, REF_COUNT));
  if (retain_graph) {
    for (size_t i = 0; i < num_forward_entries; ++i) ++ref_count[i];
  }

  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  // set output reqs
  for (size_t i = 0, j = num_forward_outputs; i < reqs.size(); ++i) {
    if (reqs[i] == kNullOp) continue;
    array_reqs[idx.entry_id(idx.outputs()[j++])] = reqs[i];
  }
  // set null reqs based on ref counts
  for (size_t i = num_forward_entries; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  const auto& mem_plan = g.GetAttr<MemoryPlanVector >(AddPrefix(BACKWARD, MEM_PLAN));
  AllocateMemory(g, idx, default_ctx, num_forward_entries, idx.num_node_entries(),
                 mem_plan, arrays, &array_reqs);

  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  RunGraph(retain_graph, idx, arrays, num_forward_nodes, idx.num_nodes(),
           std::move(array_reqs), std::move(ref_count), &states, dispatch_modes,
           Imperative::Get()->is_recording(), nullptr, monitor_callback_);

  if (retain_graph) {
    buff.resize(num_forward_entries);
  } else {
    buff.clear();
    states.clear();
  }
}

void CachedOp::StaticBackward(
    const bool retain_graph,
    const OpStatePtr& state_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  Context default_ctx = outputs[0]->ctx();

  auto& state = state_ptr.get_state<CachedOpState>();
  std::lock_guard<std::mutex> lock(state.mutex);

  bool match = SetBackwardGraph(&state.info, reqs, inputs, true);

  nnvm::Graph& g = state.info.full_graph;
  const auto& idx = g.indexed_graph();
  auto num_forward_nodes = state.info.fwd_graph.indexed_graph().num_nodes();

  if (!state.bwd_alloc || !match) {
    StaticAllocMemory(state_ptr, true, true);
  }

  // We are going to add input and output arrays to the array list.
  // The input and output arrays should only be valid for this run,
  // so we shouldn't modify the state's array list.
  state.arrays_with_in_out = state.arrays;
  auto& arrays = state.arrays_with_in_out;
  for (size_t i = 0; i < state.info.bwd_input_eid.size(); ++i) {
    auto eid = state.info.bwd_input_eid[i];
    if (eid == kEidNotExist || !state.dynamic_entries[eid]) continue;
    arrays[eid] = inputs[BwdOriginalInput(state.info.input_map, i)];
  }

  if (config_.static_shape) {
    for (auto i : config_.param_indices) {
      const auto iter = state.info.fwd_input_to_grad_output.find(i);
      if (iter == state.info.fwd_input_to_grad_output.end()) continue;
      auto entry = state.info.grad_graph.outputs[iter->second];
      if (!idx.exist(entry.node.get())) continue;
      auto eid = idx.entry_id(entry);
      if ((!arrays[eid]->IsSame(*outputs[iter->second]) &&
            state.array_reqs[eid] != kNullOp) ||
          !(state.array_reqs[eid] == reqs[iter->second])) {
        match = false;
        state.array_reqs[eid] = reqs[iter->second];
        // An input and an output may share the same array.
        INIT_DETACHED(outputs[iter->second], arrays[eid]);
        *arrays[eid] = *outputs[iter->second];
        state.dynamic_entries[eid] = false;
      }
    }
    for (auto i : config_.data_indices) {
      const auto iter = state.info.fwd_input_to_grad_output.find(i);
      if (iter == state.info.fwd_input_to_grad_output.end()) continue;
      auto entry = state.info.grad_graph.outputs[iter->second];
      if (!idx.exist(entry.node.get())) continue;
      auto eid = idx.entry_id(entry);
      state.array_reqs[eid] = reqs[iter->second];
      // An input and an output may share the same array.
      INIT_DETACHED(outputs[iter->second], arrays[eid]);
      arrays[eid] = outputs[iter->second];
    }
  } else {
    for (size_t i = 0; i < state.info.grad_graph.outputs.size(); ++i) {
      auto entry = state.info.grad_graph.outputs[i];
      if (!idx.exist(entry.node.get())) continue;
      auto eid = idx.entry_id(entry);
      state.array_reqs[eid] = reqs[i];
      // An input and an output may share the same array.
      INIT_DETACHED(outputs[i], arrays[eid]);
      arrays[eid] = outputs[i];
    }
  }

  if (!state.bwd_exec_init || !match) {
    StaticInitExec(state_ptr, true, true);
  }

  StaticRunOps(default_ctx, g, state_ptr, arrays, num_forward_nodes, idx.num_nodes());
}

void CachedOp::Backward(
    const bool retain_graph,
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  const auto& fwd_idx = fwd_graph_.indexed_graph();
  const auto& full_idx = full_graph_.indexed_graph();
  const auto& mutable_input_nodes = fwd_idx.mutable_input_nodes();
  for (size_t i = 0, j = 0; i < fwd_idx.input_nodes().size(); ++i) {
    const uint32_t nid = fwd_idx.input_nodes().at(i);
    const std::string& arg_name = fwd_idx[nid].source->attrs.name;
    const std::string profiler_scope =
        common::NodeAttrsGetProfilerScope(fwd_idx[nid].source->attrs);
    if (mutable_input_nodes.count(nid)) {
      continue;
    }
    outputs[j++]->AssignStorageInfo(profiler_scope + "arg_grad:", arg_name);
  }
  for (size_t i = fwd_idx.input_nodes().size(), j = 0;
       i < full_idx.input_nodes().size(); ++i) {
    const nnvm::NodeAttrs& attrs = full_idx[full_idx.input_nodes().at(i)].source->attrs;
    const std::string& entry_name = attrs.name;
    const std::string profiler_scope = common::NodeAttrsGetProfilerScope(attrs);
    inputs[j++]->AssignStorageInfo(profiler_scope, entry_name);
  }

  using namespace imperative;
  CHECK(!Imperative::Get()->is_recording())
      << "CachedOp does not support higher order gradients. "
      << "If you want to do backward with create_graph=True please "
      << "do not use hybridize.";

  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.backward_bulk_size);

  try {
    if (config_.static_alloc) {
      StaticBackward(retain_graph, state, inputs, reqs, outputs);
    } else {
      DynamicBackward(retain_graph, state, inputs, reqs, outputs);
    }
  } catch (const dmlc::Error& e) {
    Engine::Get()->set_bulk_size(prev_bulk_size);
    throw e;
  }

  Engine::Get()->set_bulk_size(prev_bulk_size);
}

/*
 * This is the operator state of CachedOp when CachedOp is used in the symbol
 * executor. This is different from the OpState returned by CachedOp::Forward.
 * The main reason why we need this OpState is that CachedOp and the symbol executor
 * maintain OpState differently. The symbol executor generates OpState in advance
 * while CachedOp generates OpState after Forward is called. We need this data
 * structure to keep the OpState generated by CachedOp::Forward and pass it to
 * Backward.
 */
struct CachedOpActualState {
  std::shared_ptr<CachedOp> op;
  OpStatePtr forward_state;

  explicit CachedOpActualState(std::shared_ptr<CachedOp> op) {
    this->op = op;
  }
};

/*
 * This is the forward computation when CachedOp is used as an operator in
 * a symbol executor.
 */
void CachedOpForward(const OpStatePtr& state_ptr,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  CachedOpActualState &s = state_ptr.get_state<CachedOpActualState>();
  std::vector<NDArray> in_bufs = inputs;
  std::vector<NDArray> out_bufs = outputs;
  std::vector<NDArray *> in_ptrs(in_bufs.size());
  std::vector<NDArray *> out_ptrs(out_bufs.size());
  for (size_t i = 0; i < in_ptrs.size(); i++)
    in_ptrs[i] = &in_bufs[i];
  for (size_t i = 0; i < out_ptrs.size(); i++)
    out_ptrs[i] = &out_bufs[i];

  // Set is_recording correct for the imperative executor.
  bool orig_is_record;
  if (ctx.need_grad)
    orig_is_record = Imperative::Get()->set_is_recording(true);
  else
    orig_is_record = Imperative::Get()->is_recording();
  // Set is_training correct for the imperative executor.
  bool orig_is_train;
  if (ctx.is_train)
    orig_is_train = Imperative::Get()->set_is_training(true);
  else
    orig_is_train = Imperative::Get()->is_training();
  CHECK(inputs.size() > 0) << "cached op forward requires at least 1 input";
  Context default_ctx = inputs[0].ctx();
  s.forward_state = s.op->Forward(nullptr, in_ptrs, out_ptrs, default_ctx);
  Imperative::Get()->set_is_training(orig_is_train);
  Imperative::Get()->set_is_recording(orig_is_record);
  // The arrays in out_ptrs may be changed by CachedOp.
  // If it is, we need to copy data back.
  for (size_t i = 0; i < out_bufs.size(); i++)
    if (!out_bufs[i].IsSame(outputs[i]))
      CopyFromTo(out_bufs[i], outputs[i]);
}

/*
 * This is the backward computation when CachedOp is used as an operator in
 * a symbol executor.
 */
void CachedOpBackward(const OpStatePtr& state_ptr,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  using namespace nnvm;
  using namespace imperative;
  CachedOpActualState &s = state_ptr.get_state<CachedOpActualState>();
  std::vector<NDArray> in_bufs = inputs;
  std::vector<NDArray> out_bufs = outputs;
  std::vector<NDArray *> in_ptrs;
  std::vector<NDArray *> out_ptrs;
  CHECK_EQ(s.op->num_backward_inputs(), inputs.size());
  in_ptrs.reserve(s.op->num_backward_inputs());
  out_ptrs.reserve(s.op->num_inputs());

  const std::vector<bool> &save_inputs = s.op->save_inputs();
  const std::vector<bool> &save_outputs = s.op->save_outputs();
  size_t bwd_in_dep = s.op->num_inputs();
  size_t bwd_out_dep = s.op->num_outputs();
  CHECK(s.op->num_backward_inputs() > bwd_in_dep + bwd_out_dep);
  size_t bwd_ograd_dep = s.op->num_backward_inputs() - bwd_in_dep - bwd_out_dep;

  // Find inputs, outputs and ograds
  auto ograds_begin = in_bufs.begin();
  auto ograds_end = in_bufs.begin() + bwd_ograd_dep;
  auto in_begin = ograds_end;
  auto in_end = in_begin + bwd_in_dep;
  auto out_begin = in_end;
  auto out_end = in_bufs.end();

  for (auto it = ograds_begin; it != ograds_end; it++)
    in_ptrs.push_back(&(*it));

  CHECK_EQ(save_inputs.size(), in_end - in_begin);
  CHECK_EQ(s.op->num_outputs(), out_end - out_begin);
  for (auto it = in_begin; it != in_end; it++) {
    auto i = it - in_begin;
    if (save_inputs[i])
      in_ptrs.push_back(&(*it));
  }
  for (auto it = out_begin; it != out_end; it++) {
    auto i = it - out_begin;
    if (save_outputs[i])
      in_ptrs.push_back(&(*it));
  }
  CHECK_EQ(in_ptrs.size(), s.op->num_backward_inputs());
  for (auto& out_buf : out_bufs) {
    out_ptrs.push_back(&out_buf);
  }
  CHECK_EQ(out_ptrs.size(), s.op->num_backward_outputs());
  // Set is_training correct for the imperative executor.
  bool orig_is_train;
  if (ctx.is_train)
    orig_is_train = Imperative::Get()->set_is_training(true);
  else
    orig_is_train = Imperative::Get()->is_training();
  // TODO(zhengda) CachedOp supports recording computation when running
  // the backward path. This is necessary if we want to support the second-order
  // differentiation. However, MXNet operator doesn't have an interface to
  // pass a flag to determine whether to record computation inside an operator.
  // Let's use false here for now and design a solution when the second-order
  // differentiation is supported.
  s.op->Backward(false, s.forward_state, in_ptrs, req, out_ptrs);
  Imperative::Get()->set_is_training(orig_is_train);

  // Clean up what we recorded.
  s.forward_state.reset();

  // The arrays in out_ptrs may be changed by CachedOp.
  // If it is, we need to copy data back.
  // For example, when the inputs and outputs share the same NDArrays,
  // the outputs will be replaced by inputs.
  // https://github.com/apache/incubator-mxnet/blob/v1.2.0/src/imperative/cached_op.cc#L385
  for (size_t i = 0; i < out_bufs.size(); i++)
    if (!out_bufs[i].IsSame(outputs[i]))
      CopyFromTo(out_bufs[i], outputs[i]);
}

/*
 * Register the callback to be called when the operator is executed
 */
void CachedOp::RegisterOpHook(const CachedOp::CachedOpMonCallback& callback,
                              bool monitor_all) {
    CHECK(callback) << "invalid callback";
    monitor_callback_ = callback;
    monitor_all_ = monitor_all;
}

OpStatePtr CreateCachedOpState(const NodeAttrs& attrs,
                               Context ctx,
                               const mxnet::ShapeVector& in_shapes,
                               const std::vector<int>& in_types) {
  const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
  return OpStatePtr::Create<CachedOpActualState>(op);
}

bool CachedOp::BackwardStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  using namespace imperative;
  nnvm::Graph g(full_graph_);
  const auto& idx = g.indexed_graph();
  const auto &outputs = idx.outputs();
  const size_t num_forward_outputs = fwd_graph_.outputs.size();
  CHECK_EQ(outputs.size(), num_forward_outputs + out_attrs->size());

  // Construct bwd_input_eid
  std::vector<uint32_t> bwd_input_eid;
  SetBackwardInputEid(bwd_in_dep_, bwd_out_dep_, bwd_ograd_dep_,
                      ograd_entries_, idx, &bwd_input_eid);
  CHECK_EQ(in_attrs->size(), bwd_input_eid.size());

  // Prepare stypes and contexts based on inputs
  StorageTypeVector stypes(idx.num_node_entries(), -1);
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    stypes[bwd_input_eid[i]] = in_attrs->at(i);
  }
  // Some out_attr is known ahead of time (e.g. the grad stype is given by users).
  // Prepare these to before invoking infer storage on the subgraph
  for (size_t i = 0; i < out_attrs->size(); i++) {
    const auto eid = idx.entry_id(outputs[i + num_forward_outputs]);
    if (bwd_input_eid[i] == kEidNotExist) {
      continue;
    }
    stypes[eid] = out_attrs->at(i);
  }
  exec::DevMaskVector dev_masks(idx.num_nodes(), dev_mask);

  // Full graph storage type inference
  CheckAndInferStorageType(&g, std::move(dev_masks), std::move(stypes), false);
  // Retrieve result and set outputs
  const auto& inferred_stypes = g.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = 0; i < out_attrs->size(); i++) {
    const auto eid = idx.entry_id(outputs[i + num_forward_outputs]);
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, inferred_stypes[eid]);
  }
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

void CachedOpParamParser(nnvm::NodeAttrs* attrs) {
  CachedOpConfig param;
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
  if (!param.subgraph.empty()) {
    nnvm::Graph g = nnvm::pass::LoadJSON(param.subgraph);
    CHECK(!g.outputs.empty());
    nnvm::Symbol sym;
    sym.outputs = g.outputs;
    std::vector<std::pair<std::string, std::string> > flags;
    for (const auto& attr : attrs->dict)
      flags.emplace_back(attr.first, attr.second);
    attrs->parsed = std::make_shared<CachedOp>(sym, flags);
  }
}

size_t CachedOp::BwdOriginalInput(const std::vector<size_t>& input_map, size_t new_i) {
  CHECK_GE(input_map.size(), bwd_in_dep_.size());
  if (new_i >= bwd_ograd_dep_.size() && new_i < bwd_ograd_dep_.size() + bwd_in_dep_.size())
    return bwd_ograd_dep_.size() + input_map[new_i - bwd_ograd_dep_.size()];
  return new_i;
}

NNVM_REGISTER_OP(_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr_parser(CachedOpParamParser)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(n->attrs.parsed);
    return op->Gradient(n, ograds);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->ListForwardInputNames();
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->ListForwardOutputNames();
  })
.set_attr<FCreateOpState>("FCreateOpState", CreateCachedOpState)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_shapes,
     mxnet::ShapeVector *out_shapes) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op::DefaultSubgraphOpShapeHelper(op->GetForwardSym(), in_shapes, out_shapes);
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_types,
     std::vector<int> *out_types) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op::DefaultSubgraphOpTypeHelper(op->GetForwardSym(), in_types, out_types);
  })
.set_attr<FInferStorageType>("FInferStorageType",
  [](const nnvm::NodeAttrs& attrs,
     const int dev_mask,
     DispatchMode* dispatch_mode,
     std::vector<int>* in_stypes,
     std::vector<int>* out_stypes) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op::DefaultSubgraphOpStorageTypeHelper(op->GetForwardSym(),
                                                  dev_mask, dispatch_mode,
                                                  in_stypes, out_stypes);
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CachedOpForward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CachedOpForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op::DefaultSubgraphOpMutableInputsHelper(op->GetForwardSym());
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op::DefaultSubgraphOpResourceRequestHelper(op->GetForwardSym());
  })
.set_attr<FExecType>("FExecType", op::DefaultSubgraphOpExecType)
.add_argument("data", "NDArray-or-Symbol[]", "input data list");

NNVM_REGISTER_OP(_backward_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_backward_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs() - op->mutable_input_nodes().size();
  })
.set_attr<FInferStorageType>("FInferStorageType", [](const nnvm::NodeAttrs& attrs,
                                                     const int dev_mask,
                                                     DispatchMode* dispatch_mode,
                                                     std::vector<int> *in_attrs,
                                                     std::vector<int> *out_attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->BackwardStorageType(attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CachedOpBackward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CachedOpBackward)
.set_attr<FExecType>("FExecType", op::DefaultSubgraphOpExecType)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<bool>("TIsBackward", true);

}  // namespace mxnet
