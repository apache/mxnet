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

DMLC_REGISTER_PARAMETER(CachedOpParam);

Imperative::CachedOp::CachedOp(
    const nnvm::Symbol& sym,
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const auto _copy = Op::Get("_copy");

  param_.Init(kwargs);

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

    inlining_ = (idx.num_nodes() - idx.input_nodes().size()) <= param_.inline_limit;
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
    curr_grad_req_ = std::vector<bool>(grad_graph_.outputs.size(), true);
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

nnvm::Graph Imperative::CachedOp::GetForwardGraph(
    const bool recording, const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(inputs.size(), num_inputs());
  nnvm::Graph& g = fwd_graph_;

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
    return g;
  }

  const auto& idx = g.indexed_graph();

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  for (const auto i : idx.input_nodes()) storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >(
          recording ? "full_ref_count" : "forward_ref_count"));
  g.attrs[recording ? "full_mem_plan" : "forward_mem_plan"] =
      std::make_shared<dmlc::any>(std::move(mem_plan));

  return g;
}

nnvm::Graph Imperative::CachedOp::GetBackwardGraph(
    const OpStatePtr& op_state,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  nnvm::Graph& g = full_graph_;
  auto& state = op_state.get_state<CachedOpState>();
  bool req_match = true;
  for (size_t i = 0; i < reqs.size(); ++i) {
    if (curr_grad_req_[i] != (reqs[i] != kNullOp)) {
      curr_grad_req_[i] = reqs[i] != kNullOp;
      req_match = false;
    }
  }
  if (!req_match) {
    g = nnvm::Graph();
    g.outputs = fwd_graph_.outputs;
    for (size_t i = 0; i < grad_graph_.outputs.size(); ++i) {
      if (curr_grad_req_[i]) g.outputs.emplace_back(grad_graph_.outputs[i]);
    }
    bwd_input_eid_.clear();
  }

  const auto& idx = g.indexed_graph();

  if (bwd_input_eid_.size() != inputs.size()) {
    bwd_input_eid_.clear();
    for (const auto& i : bwd_ograd_dep_) {
      auto eid = idx.entry_id(ograd_entries_[i]);
      bwd_input_eid_.push_back(eid);
    }
    for (const auto& i : bwd_in_dep_) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      bwd_input_eid_.push_back(eid);
    }
    for (const auto& i : bwd_out_dep_) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      bwd_input_eid_.push_back(eid);
    }
    CHECK_EQ(inputs.size(), bwd_input_eid_.size());
  }

  size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();

  if (!g.attrs.count("backward_ref_count")) {
    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }
    for (size_t i = 0; i < inputs.size(); ++i) ++ref_count[bwd_input_eid_[i]];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    g.attrs["backward_ref_count"] = std::make_shared<dmlc::any>(std::move(ref_count));
  }

  ShapeVector shapes(idx.num_node_entries(), TShape());
  DTypeVector dtypes(idx.num_node_entries(), -1);
  StorageTypeVector stypes(idx.num_node_entries(), -1);

  for (size_t i = 0; i < num_forward_entries; ++i) {
    shapes[i] = state.buff[i].shape();
    dtypes[i] = state.buff[i].dtype();
    stypes[i] = state.buff[i].storage_type();
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    shapes[bwd_input_eid_[i]] = inputs[i]->shape();
    dtypes[bwd_input_eid_[i]] = inputs[i]->dtype();
    stypes[bwd_input_eid_[i]] = inputs[i]->storage_type();
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
    return g;
  }

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  for (size_t i = 0; i < num_forward_entries; ++i) storage[i] = exec::kExternalStorageID;
  for (const auto i : idx.input_nodes()) storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  for (const auto i : idx.outputs()) storage[idx.entry_id(i)] = exec::kExternalStorageID;

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >("backward_ref_count"),
      {num_forward_nodes, idx.num_nodes()}, {num_forward_entries, idx.num_node_entries()});
  g.attrs["backward_mem_plan"] = std::make_shared<dmlc::any>(std::move(mem_plan));

  return g;
}

void Imperative::CachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;
  static const auto cached_op = nnvm::Op::Get("_CachedOp");

  // Initialize
  bool recording = Imperative::Get()->is_recording();
  nnvm::Graph g = GetForwardGraph(recording, inputs);
  const auto& idx = g.indexed_graph();
  size_t num_inputs = idx.input_nodes().size();

  CHECK_EQ(num_inputs, inputs.size())
      << "CachedOp requires " << num_inputs << " but got " << inputs.size();

  Context default_ctx = inputs[0]->ctx();
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name << " is on " << default_ctx
        << " while " << idx[idx.input_nodes()[i]].source->attrs.name << " is on "
        << inputs[i]->ctx();
  }

  auto op_state_ptr = OpStatePtr::Create<CachedOpState>();
  auto& cached_op_state = op_state_ptr.get_state<CachedOpState>();
  auto& buff = cached_op_state.buff;
  auto& states = cached_op_state.states;

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
  int prev_bulk_size = Engine::Get()->set_bulk_size(param_.forward_bulk_size);

  Imperative::Get()->RunGraph(
      false, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
      std::move(ref_count), &states, dispatch_modes);

  Engine::Get()->set_bulk_size(prev_bulk_size);
  Imperative::Get()->set_is_recording(recording);

  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (arrays[i] == &buff[i]) continue;
    buff[i].shape_ = arrays[i]->shape_;
    buff[i].dtype_ = arrays[i]->dtype_;
    buff[i].storage_type_ = arrays[i]->storage_type_;
  }

  if (recording && !inlining_) {
    nnvm::NodeAttrs attrs;
    attrs.op = cached_op;
    attrs.name = "_cachedop";
    attrs.parsed = op_ptr;
    Imperative::Get()->RecordOp(
        std::move(attrs), inputs, outputs, op_state_ptr,
        &save_inputs(), &save_outputs());
  }
}


void Imperative::CachedOp::Backward(
    const bool retain_graph,
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;
  CHECK(!Imperative::Get()->is_recording())
      << "CachedOp does not support higher order gradients. "
      << "If you want to do backward with create_graph=True please "
      << "do not use hybridize.";

  // Initialize
  nnvm::Graph g = GetBackwardGraph(state, reqs, inputs);
  const auto& idx = g.indexed_graph();

  auto& cached_op_state = state.get_state<CachedOpState>();
  auto& buff = cached_op_state.buff;
  auto& states = cached_op_state.states;

  size_t num_forward_outputs = fwd_graph_.outputs.size();
  size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();
  buff.resize(idx.num_node_entries());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  for (size_t i = 0; i < inputs.size(); ++i) {
    arrays[bwd_input_eid_[i]] = inputs[i];
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
  for (size_t i = num_forward_entries; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  Context default_ctx = outputs[0]->ctx();
  const auto& mem_plan = g.GetAttr<MemoryPlanVector >("backward_mem_plan");
  AllocateMemory(g, idx, default_ctx, num_forward_entries, idx.num_node_entries(),
                 mem_plan, arrays, &array_reqs);

  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  int prev_bulk_size = Engine::Get()->set_bulk_size(param_.backward_bulk_size);

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
