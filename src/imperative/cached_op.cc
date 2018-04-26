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

namespace mxnet {

DMLC_REGISTER_PARAMETER(CachedOpConfig);

Imperative::CachedOp::CachedOp(
    const nnvm::Symbol& sym,
    const std::vector<std::pair<std::string, std::string> >& flags,
    const std::vector<std::string> arg_names,
    const std::unordered_map<std::string, std::vector<NDArray> >& params) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const auto _copy = Op::Get("_copy");

  config_.Init(flags);

  // construct forward graph
  {
    NodeEntryMap<int> dedup_out;
    for (const auto& i : sym.outputs) {
      if (dedup_out.count(i)) {
        NodePtr copy_node = Node::Create();
        copy_node->attrs.op = _copy;
        copy_node->attrs.name =
            i.node->attrs.name + "_copy" + std::to_string(dedup_out[i]++);
        copy_node->inputs.emplace_back(i);
        if (_copy->attr_parser != nullptr) {
          _copy->attr_parser(&(copy_node->attrs));
        }
        fwd_graph_.outputs.push_back(NodeEntry{copy_node, 0, 0});
      } else {
        dedup_out.insert({i, 0});
        fwd_graph_.outputs.push_back(i);
      }
    }
    const auto& idx = fwd_graph_.indexed_graph();
    CHECK_GE(idx.input_nodes().size(), 1) << "CachedOp requires at least 1 input";

    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (const auto& i : idx.input_nodes()) ++ref_count[idx.entry_id(i, 0)];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    for (size_t i = 0; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }

    fwd_graph_.attrs["forward_ref_count"] =
        std::make_shared<dmlc::any>(std::move(ref_count));

    inlining_ = (idx.num_nodes() - idx.input_nodes().size()) <= config_.inline_limit;
  }

  // Set params
  {
    const auto& idx = fwd_graph_.indexed_graph();
    std::unordered_map<std::string, size_t> arg_name_to_id;
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      const auto& name = idx[idx.input_nodes()[i]].source->attrs.name;
      auto iter = params.find(name);
      if (iter == params.end()) {
        arg_name_to_id[name] = i;
        continue;
      }
      fwd_params_idx_.push_back(i);
      for (const auto& param : iter->second) {
        params_[param.ctx()].emplace_back(param);
      }
    }

    CHECK_EQ(arg_name_to_id.size(), arg_names.size())
        << "CachedOp expects " << arg_name_to_id.size()
        << " inputs, given " << arg_names.size();

    for (const auto& name : arg_names) {
      auto iter = arg_name_to_id.find(name);
      CHECK(iter != arg_name_to_id.end()) << "Unexpected input name " << name;
      fwd_args_idx_.push_back(iter->second);
    }
  }

  // construct backward graph
  {
    ograd_entries_.reserve(fwd_graph_.outputs.size());
    for (size_t i = 0; i < fwd_graph_.outputs.size(); ++i) {
      ograd_entries_.emplace_back(NodeEntry{Node::Create(), 0, 0});
    }

    std::vector<NodeEntry> xs;
    std::vector<NodePtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    for (const auto& i : args) xs.emplace_back(NodeEntry{i, 0, 0});
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";

    grad_graph_ = pass::Gradient(
        fwd_graph_, fwd_graph_.outputs, xs, ograd_entries_,
        exec::AggregateGradient, nullptr, nullptr,
        zero_ops, "_copy");
  }

  // construct full graph
  {
    size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
    size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();

    full_graph_.outputs = fwd_graph_.outputs;
    bwd_grad_req_ = std::vector<bool>(grad_graph_.outputs.size(), true);
    for (const auto& i : grad_graph_.outputs) full_graph_.outputs.emplace_back(i);
    const auto& idx = full_graph_.indexed_graph();

    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) {
         ++ref_count[idx.entry_id(j)];
      }
    }

    auto full_ref_count = fwd_graph_.GetAttr<std::vector<uint32_t> >("forward_ref_count");
    for (size_t i = 0; i < num_forward_entries; ++i) full_ref_count[i] += ref_count[i];
    fwd_graph_.attrs["full_ref_count"] =
        std::make_shared<dmlc::any>(std::move(full_ref_count));

    size_t num_forward_inputs = num_inputs();
    size_t num_forward_outputs = num_outputs();
    for (uint32_t i = 0; i < ograd_entries_.size(); ++i) {
      if (!idx.exist(ograd_entries_[i].node.get())) continue;
      auto eid = idx.entry_id(ograd_entries_[i]);
      if (ref_count[eid] > 0) {
        bwd_ograd_dep_.push_back(i);
      }
    }
    save_inputs_.resize(num_forward_inputs, false);
    for (uint32_t i = 0; i < num_forward_inputs; ++i) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      if (ref_count[eid] > 0) {
        save_inputs_[i] = true;
        bwd_in_dep_.push_back(i);
      }
    }
    save_outputs_.resize(idx.outputs().size(), false);
    for (uint32_t i = 0; i < num_forward_outputs; ++i) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      if (ref_count[eid] > 0) {
        save_outputs_[i] = true;
        bwd_out_dep_.push_back(i);
      }
    }
  }
}

std::vector<nnvm::NodeEntry> Imperative::CachedOp::Gradient(
    const nnvm::NodePtr& node,
    const std::vector<nnvm::NodeEntry>& ograds) {
  using namespace nnvm;
  static const auto _backward_CachedOp = Op::Get("_backward_CachedOp");
  static const auto _NoGrad = Op::Get("_NoGradient");

  auto p = Node::Create();
  p->attrs.op = _backward_CachedOp;
  p->attrs.name = node->attrs.name + "_backward";
  p->attrs.parsed = node->attrs.parsed;
  p->control_deps.push_back(node);
  p->inputs.reserve(bwd_ograd_dep_.size() + bwd_in_dep_.size() + bwd_out_dep_.size());
  for (auto i : bwd_ograd_dep_) p->inputs.push_back(ograds[i]);
  for (auto i : bwd_in_dep_) p->inputs.push_back(node->inputs[i]);
  for (auto i : bwd_out_dep_) p->inputs.emplace_back(NodeEntry{node, i, 0});
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
        ret.emplace_back(NodeEntry{nop, 0, 0});
      } else {
        ret.emplace_back(NodeEntry{p, k++, 0});
      }
    }
  } else {
    for (uint32_t i = 0; i < num_inputs(); ++i) ret.emplace_back(NodeEntry{p, i, 0});
  }
  return ret;
}


bool Imperative::CachedOp::SetForwardGraph(
    GraphInfo* info,
    const bool recording,
    const std::vector<NDArray*>& inputs) {
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
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    shape_inputs.emplace_back(inputs[i]->shape());
    dtype_inputs.emplace_back(inputs[i]->dtype());
    storage_type_inputs.emplace_back(inputs[i]->storage_type());
  }

  bool match = true;
  match &= CheckAndInferShape(&g, std::move(shape_inputs), true);
  match &= CheckAndInferType(&g, std::move(dtype_inputs), true);
  exec::DevMaskVector dev_mask(g.indexed_graph().num_nodes(), inputs[0]->ctx().dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask),
                                    std::move(storage_type_inputs), true);

  if (!match) {
    g.attrs.erase("forward_mem_plan");
    g.attrs.erase("full_mem_plan");
  } else if (g.attrs.count(recording ? "full_mem_plan" : "forward_mem_plan")) {
    return true;
  }

  const auto& idx = g.indexed_graph();

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  for (const auto i : idx.input_nodes()) storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  CHECK_EQ(stypes.size(), storage.size());
  for (size_t i = 0; i < stypes.size(); i++) {
    if (stypes[i] != kDefaultStorage)
      storage[i] = exec::kDynamicStorageID;
  }

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >(
          recording ? "full_ref_count" : "forward_ref_count"));
  g.attrs[recording ? "full_mem_plan" : "forward_mem_plan"] =
      std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}

bool Imperative::CachedOp::SetBackwardGraph(
    GraphInfo* info,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  nnvm::Graph& g = info->full_graph;
  bool req_match = true;
  for (size_t i = 0; i < reqs.size(); ++i) {
    if (info->bwd_grad_req[i] != (reqs[i] != kNullOp)) {
      info->bwd_grad_req[i] = reqs[i] != kNullOp;
      req_match = false;
    }
  }
  if (!req_match) {
    g = nnvm::Graph();
    g.outputs = fwd_graph_.outputs;
    for (size_t i = 0; i < grad_graph_.outputs.size(); ++i) {
      if (info->bwd_grad_req[i]) g.outputs.emplace_back(grad_graph_.outputs[i]);
    }
    info->bwd_input_eid.clear();
  }

  const auto& idx = g.indexed_graph();

  if (info->bwd_input_eid.size() != inputs.size()) {
    info->bwd_input_eid.clear();
    for (const auto& i : bwd_ograd_dep_) {
      auto eid = idx.entry_id(ograd_entries_[i]);
      info->bwd_input_eid.push_back(eid);
    }
    for (const auto& i : bwd_in_dep_) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      info->bwd_input_eid.push_back(eid);
    }
    for (const auto& i : bwd_out_dep_) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      info->bwd_input_eid.push_back(eid);
    }
    CHECK_EQ(inputs.size(), info->bwd_input_eid.size());
  }

  size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();

  if (!g.attrs.count("backward_ref_count")) {
    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }
    for (size_t i = 0; i < inputs.size(); ++i) ++ref_count[info->bwd_input_eid[i]];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    g.attrs["backward_ref_count"] = std::make_shared<dmlc::any>(std::move(ref_count));
  }

  auto shapes = info->fwd_graph.GetAttr<ShapeVector>("shape");
  shapes.resize(idx.num_node_entries(), TShape());
  auto dtypes = info->fwd_graph.GetAttr<DTypeVector>("dtype");
  dtypes.resize(idx.num_node_entries(), -1);
  auto stypes = info->fwd_graph.GetAttr<StorageTypeVector>("storage_type");
  stypes.resize(idx.num_node_entries(), -1);

  for (size_t i = 0; i < inputs.size(); ++i) {
    shapes[info->bwd_input_eid[i]] = inputs[i]->shape();
    dtypes[info->bwd_input_eid[i]] = inputs[i]->dtype();
    stypes[info->bwd_input_eid[i]] = inputs[i]->storage_type();
  }

  std::pair<uint32_t, uint32_t> node_range, entry_range;
  node_range = {num_forward_nodes, idx.num_nodes()};
  entry_range = {num_forward_entries, idx.num_node_entries()};

  bool match = true;
  match &= CheckAndInferShape(&g, std::move(shapes), false,
                              node_range, entry_range);
  match &= CheckAndInferType(&g, std::move(dtypes), false,
                             node_range, entry_range);
  exec::DevMaskVector dev_mask(idx.num_nodes(), inputs[0]->ctx().dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask), std::move(stypes),
                                    false, node_range, entry_range);

  if (!match) {
    g.attrs.erase("backward_mem_plan");
  } else if (g.attrs.count("backward_mem_plan")) {
    return true;
  }

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  for (size_t i = 0; i < num_forward_entries; ++i) storage[i] = exec::kExternalStorageID;
  for (const auto i : idx.input_nodes()) storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  for (const auto i : idx.outputs()) storage[idx.entry_id(i)] = exec::kExternalStorageID;
  for (size_t i = 0; i < stypes.size(); i++) {
    if (stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
  }

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >("backward_ref_count"),
      {num_forward_nodes, idx.num_nodes()}, {num_forward_entries, idx.num_node_entries()});
  g.attrs["backward_mem_plan"] = std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}


OpStatePtr Imperative::CachedOp::StaticForward(
    const Context& default_ctx,
    const std::vector<NDArray*>& args,
    const std::vector<NDArray*>& outputs) {
}


OpStatePtr Imperative::CachedOp::DynamicForward(
    const Context& default_ctx,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  // Initialize
  bool recording = Imperative::Get()->is_recording();
  auto op_state = OpStatePtr::Create<DynamicRuntime>();
  auto& runtime = op_state.get_state<DynamicRuntime>();
  {
    auto dev_state = GetDeviceState(default_ctx);
    std::lock_guard<std::mutex> lock(dev_state->mutex);
    SetForwardGraph(&dev_state->info, recording, inputs);
    runtime.info.fwd_graph = dev_state->info.fwd_graph;
  }
  nnvm::Graph& g = runtime.info.fwd_graph;
  const auto& idx = g.indexed_graph();
  size_t num_inputs = idx.input_nodes().size();
  auto& buff = runtime.buff;
  auto& states = runtime.op_states;

  // Allocate entries
  states.resize(idx.num_nodes());
  buff.resize(idx.num_node_entries());
  states.reserve(idx.num_nodes());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  for (size_t i = 0; i < num_inputs; ++i) {
    arrays[idx.entry_id(idx.input_nodes()[i], 0)] = inputs[i];
  }
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    if (!arrays[eid]->is_none()) *outputs[i] = arrays[eid]->Detach();
    arrays[eid] = outputs[i];
  }

  // Allocate NDArrays
  std::vector<uint32_t> ref_count = g.GetAttr<std::vector<uint32_t> >(
      recording ? "full_ref_count" : "forward_ref_count");

  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  const auto& mem_plan = g.GetAttr<MemoryPlanVector >(
      recording ? "full_mem_plan" : "forward_mem_plan");
  AllocateMemory(g, idx, default_ctx, 0, idx.num_node_entries(),
                 mem_plan, arrays, &array_reqs);

  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  if (recording && !inlining_) Imperative::Get()->set_is_recording(false);
  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.forward_bulk_size);

  Imperative::Get()->RunGraph(
      false, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
      std::move(ref_count), &states, dispatch_modes);

  Engine::Get()->set_bulk_size(prev_bulk_size);
  Imperative::Get()->set_is_recording(recording);

  return op_state;
}

void Imperative::CachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& args,
    const std::vector<NDArray*>& outputs) {
  static const auto cached_op = nnvm::Op::Get("_CachedOp");

  CHECK_EQ(args.size(), fwd_args_idx_.size())
      << "CachedOp requires " << fwd_args_idx_.size()
      << " inputs but got " << args.size();

  Context default_ctx = args[0]->ctx();

  const auto& idx = fwd_graph_.indexed_graph();
  for (size_t i = 0; i < fwd_args_idx_.size(); ++i) {
    CHECK_EQ(args[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[fwd_args_idx_[0]]].source->attrs.name
        << " is on " << default_ctx << " while "
        << idx[idx.input_nodes()[fwd_args_idx_[i]]].source->attrs.name
        << " is on " << args[i]->ctx();
  }

  std::vector<NDArray*> inputs(num_inputs());
  for (index_t i = 0; i < fwd_args_idx_.size(); ++i) {
    inputs[fwd_args_idx_[i]] = args[i];
  }
  if (fwd_params_idx_.size()) {
    CHECK(params_.find(default_ctx) != params_.end())
        << "CachedOp is not initialized on context " << default_ctx;

    for (size_t i = 0; i < fwd_params_idx_.size(); ++i) {
      inputs[fwd_params_idx_[i]] = &params_[default_ctx][i];
    }
  }

  OpStatePtr op_state;
  if (config_.use_static_memory) {
    op_state = StaticForward(default_ctx, args, outputs);
  } else {
    op_state = DynamicForward(default_ctx, inputs, outputs);
  }

  if (Imperative::Get()->is_recording() && !inlining_) {
    nnvm::NodeAttrs attrs;
    attrs.op = cached_op;
    attrs.name = "_cachedop";
    attrs.parsed = op_ptr;
    Imperative::Get()->RecordOp(
        std::move(attrs), inputs, outputs, op_state,
        &save_inputs(), &save_outputs());
  }
}


void Imperative::CachedOp::DynamicBackward(
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
    auto dev_state = GetDeviceState(default_ctx);
    std::lock_guard<std::mutex> lock(dev_state->mutex);
    dev_state->info.fwd_graph = runtime.info.fwd_graph;
    SetBackwardGraph(&dev_state->info, reqs, inputs);
    runtime.info.full_graph = dev_state->info.full_graph;
    runtime.info.bwd_input_eid = dev_state->info.bwd_input_eid;
  }
  nnvm::Graph& g = runtime.info.full_graph;
  const auto& idx = g.indexed_graph();
  auto& buff = runtime.buff;
  auto& states = runtime.op_states;

  size_t num_forward_outputs = fwd_graph_.outputs.size();
  size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();
  buff.resize(idx.num_node_entries());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  for (size_t i = 0; i < inputs.size(); ++i) {
    arrays[runtime.info.bwd_input_eid[i]] = inputs[i];
  }
  for (size_t i = 0, j = num_forward_outputs; i < reqs.size(); ++i) {
    if (reqs[i] == kNullOp) continue;
    arrays[idx.entry_id(idx.outputs()[j++])] = outputs[i];
  }

  // Allocate NDArrays
  auto ref_count = g.GetAttr<std::vector<uint32_t> >("backward_ref_count");
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

  const auto& mem_plan = g.GetAttr<MemoryPlanVector >("backward_mem_plan");
  AllocateMemory(g, idx, default_ctx, num_forward_entries, idx.num_node_entries(),
                 mem_plan, arrays, &array_reqs);

  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.backward_bulk_size);

  Imperative::Get()->RunGraph(
      retain_graph, idx, arrays, num_forward_nodes, idx.num_nodes(),
      std::move(array_reqs), std::move(ref_count), &states, dispatch_modes);

  Engine::Get()->set_bulk_size(prev_bulk_size);

  if (retain_graph) {
    buff.resize(num_forward_entries);
  } else {
    buff.clear();
    states.clear();
  }
}

void Imperative::CachedOp::StaticBackward(
    const bool retain_graph,
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
}

void Imperative::CachedOp::Backward(
    const bool retain_graph,
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace imperative;
  CHECK(!Imperative::Get()->is_recording())
      << "CachedOp does not support higher order gradients. "
      << "If you want to do backward with create_graph=True please "
      << "do not use hybridize.";

  if (config_.use_static_memory) {
    StaticBackward(retain_graph, state, inputs, reqs, outputs);
  } else {
    DynamicBackward(retain_graph, state, inputs, reqs, outputs);
  }
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
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(n->attrs.parsed);
    return op->Gradient(n, ograds);
  });

NNVM_REGISTER_OP(_backward_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_backward_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs() - op->mutable_input_nodes().size();
  })
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<bool>("TIsBackward", true);

}  // namespace mxnet
