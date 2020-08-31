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
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "./imperative_utils.h"
#include "./cached_op.h"

namespace nnvm {
ObjectPtr CreateVariableNode(const std::string &name);
}

namespace mxnet {
#if DMLC_CXX11_THREAD_LOCAL
thread_local bool Imperative::is_train_ = false;
thread_local bool Imperative::is_recording_ = false;
thread_local bool Imperative::is_deferred_compute_ = false;
thread_local bool Imperative::is_np_shape_thread_local_ = false;
#else
MX_THREAD_LOCAL bool Imperative::is_train_ = false;
MX_THREAD_LOCAL bool Imperative::is_recording_ = false;
MX_THREAD_LOCAL bool Imperative::is_deferred_compute_ = false;
MX_THREAD_LOCAL bool Imperative::is_np_shape_thread_local_ = false;
#endif

Imperative* Imperative::Get() {
  static Imperative inst;
  return &inst;
}

OpStatePtr Imperative::InvokeOp(
    const Context& ctx,
    const nnvm::NodeAttrs& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const std::vector<OpReqType>& req,
    const DispatchMode dispatch_mode,
    OpStatePtr state) {
  using namespace imperative;
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();

  const nnvm::Op *op = attrs.op;

  std::vector<engine::VarHandle> read_vars, write_vars;
  std::vector<Resource> requested;
  std::vector<uint32_t> mutate_idx;
  SetDependency(attrs, ctx, inputs, outputs,
      &read_vars, &write_vars, &requested, &mutate_idx, dispatch_mode);

  FCompute fn = common::GetFCompute<FCompute>(op, "FCompute", ctx);
  FComputeEx fn_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", ctx);

  // FComputeEx is dispatched only when dispatch_mode is DispatchMode::kFComputeEx
  CHECK(dispatch_mode != DispatchMode::kUndefined);
  bool dispatch_fcompex = dispatch_mode == DispatchMode::kFComputeEx;
  if (fn_ex && dispatch_fcompex) {
    PushFComputeEx(fn_ex, op, attrs, ctx, read_vars, write_vars,
        requested, inputs, outputs, req);
  } else if (fn) {
    PushFCompute(fn, op, attrs, ctx, read_vars, write_vars,
        requested, inputs, outputs, mutate_idx, req);
  } else if (createop.count(op) || is_layer_backward.get(op, false)) {
    if (!state) {
      state = createop[op](attrs, ctx, ret->arg_shapes, ret->arg_types);
    }
    write_vars.push_back(state.get_var());
    PushOperator(state, op, attrs, ctx, read_vars, write_vars,
        requested, inputs, outputs, mutate_idx, req, dispatch_mode);
  } else {
    LOG(FATAL)
      << "Operator " << op->name << " is not implemented for "
      << (ctx.dev_mask() == gpu::kDevMask ? "GPU." : "CPU.");
  }

  return state;
}

OpStatePtr Imperative::Invoke(
    const Context& default_ctx,
    const nnvm::NodeAttrs& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace imperative;
  static auto& ndfunc = nnvm::Op::GetAttr<FNDArrayFunction>("FNDArrayFunction");

  if (ndfunc.count(attrs.op)) {
    std::vector<NDArray> p_inputs, p_outputs;
    DerefInputOutput(inputs, outputs, &p_inputs, &p_outputs);
    ndfunc[attrs.op](attrs, p_inputs, &p_outputs);
    for (size_t i = 0; i < outputs.size(); ++i) *outputs[i] = std::move(p_outputs[i]);
    return OpStatePtr();
  }

  // TODO(piiswrong): infer ctx
  DispatchMode dispatch_mode = DispatchMode::kUndefined;
  Context ctx = GetContext(attrs, inputs, outputs, default_ctx);
  SetShapeType(ctx, attrs, inputs, outputs, &dispatch_mode);
  std::vector<OpReqType> req;
  SetWriteInplaceReq(inputs, outputs, &req);
  OpStatePtr ret = InvokeOp(ctx, attrs, inputs, outputs, req, dispatch_mode);
  // the followinng loop is used for finding out the correct shape when some shapes are dynamic
  for (auto output : outputs) {
    if (!shape_is_known(output->shape())) {
      // the WaitToRead overhead here does not seem to be avoidable
      output->WaitToRead();
      output->SetShapeFromChunk();
    }
  }
  return ret;
}

// Create nnvm::NodeEntry for variables' and gradients' autograd_entry_
// attribute and associate AGInfo with it's info attribute
void Imperative::MarkVariables(
    const std::vector<NDArray*>& variables,
    const std::vector<uint32_t>& grad_reqs,
    const std::vector<NDArray*>& gradients) {
  for (uint32_t i = 0; i < variables.size(); ++i) {
    std::string str_c(std::to_string(variable_count_++));

    variables[i]->autograd_entry_ = nnvm::NodeEntry{
        nnvm::Symbol::CreateVariable("var" + str_c).outputs[0].node, 0, 0};
    AGInfo& info = AGInfo::Create(variables[i]->autograd_entry_.node);
    info.outputs.emplace_back(variables[i]->Detach());
    info.out_grads.emplace_back(gradients[i]->Detach());
    info.grad_req = static_cast<OpReqType>(grad_reqs[i]);
    info.ctx = variables[i]->ctx();

    gradients[i]->autograd_entry_ = nnvm::NodeEntry{
        nnvm::Symbol::CreateVariable("grad" + str_c).outputs[0].node, 0, 0};
    AGInfo& grad_info = AGInfo::Create(gradients[i]->autograd_entry_.node);
    grad_info.outputs.emplace_back(gradients[i]->Detach());
    grad_info.ctx = gradients[i]->ctx();
  }
}


void Imperative::GetBackwardDependency(
    const nnvm::ObjectPtr& node,
    uint32_t num_inputs, uint32_t num_outputs,
    std::vector<bool> *p_save_inputs,
    std::vector<bool> *p_save_outputs) {
  static auto& fgradient = nnvm::Op::GetAttr<nnvm::FGradient>("FGradient");
  std::vector<bool>& save_inputs = *p_save_inputs;
  std::vector<bool>& save_outputs = *p_save_outputs;
  save_inputs.resize(num_inputs);
  save_outputs.resize(num_outputs);
  std::fill(save_inputs.begin(), save_inputs.end(), false);
  std::fill(save_outputs.begin(), save_outputs.end(), false);

  node->inputs.clear();
  node->inputs.reserve(num_inputs);
  for (uint32_t i = 0; i < num_inputs; ++i) {
    node->inputs.emplace_back(nnvm::NodeEntry{nullptr, i, 0});
  }

  if (fgradient.count(node->op())) {
    std::vector<nnvm::NodeEntry> ograd_entries;
    ograd_entries.reserve(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i) {
      ograd_entries.emplace_back(nullptr, i, 1);
    }
    auto igrad_entries = fgradient[node->op()](node, ograd_entries);
    for (const auto& i : igrad_entries) {
      if (i.node == nullptr && i.version == 0) {
        save_inputs[i.index] = true;
      } else if (i.node == node) {
        save_outputs[i.index] = true;
      }
    }
    DFSVisit(igrad_entries, [&](const nnvm::ObjectPtr& gnode) {
        if (!gnode || gnode == node) return;
        for (const auto& i : gnode->inputs) {
          if (i.node == nullptr && i.version == 0) {
            save_inputs[i.index] = true;
          } else if (i.node == node) {
            save_outputs[i.index] = true;
          }
        }
      });
  }
}

void Imperative::RecordOp(
    nnvm::NodeAttrs&& attrs,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const OpStatePtr& state,
    std::vector<bool>* p_save_inputs,
    std::vector<bool>* p_save_outputs) {
  MXAPIThreadLocalEntry<> *local_buff = MXAPIThreadLocalStore<>::Get();

  CHECK(!is_deferred_compute())
      << "Autograd recording is not supported during deferred compute mode.";

  for (auto output : outputs) {
    CHECK(AGInfo::IsNone(*output))
      << "Assigning to NDArrays that are already in a computational graph "
      << "will cause undefined behavior when evaluating gradients. "
      << "Please call backward first to clear the graph or do this out side of "
      << "a record section. Also note that you cannot use inplace operations "
      << "like +=, *=, relu(x, out=x), y[idx]=x, etc inside a record section. "
      << "Issue occurred while recording op: " << attrs.name;
  }

  bool need_grad = false;
  for (const auto& i : inputs) {
    if (AGInfo::IsNone(*i)) continue;
    need_grad = true;
    break;
  }
  if (!need_grad) return;

  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs = std::move(attrs);
  node->attrs.name = "node_" + std::to_string(node_count_++);
  AGInfo& info = AGInfo::Create(node);
  info.state = state;
  info.ctx = outputs[0]->ctx();

  if (p_save_inputs == nullptr) {
    p_save_inputs = &(local_buff->save_inputs);
    p_save_outputs = &(local_buff->save_outputs);
    GetBackwardDependency(
        node, inputs.size(), outputs.size(), p_save_inputs, p_save_outputs);
  } else {
    node->inputs.resize(inputs.size());
  }

  std::vector<bool>& save_inputs = *p_save_inputs;
  std::vector<bool>& save_outputs = *p_save_outputs;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (AGInfo::IsNone(*(inputs[i]))) {
      nnvm::NodeEntry entry{nnvm::Symbol::CreateVariable(
          "null" + std::to_string(variable_count_++)).outputs[0].node, 0, 0};
      AGInfo& input_info = AGInfo::Create(entry.node);
      input_info.ctx = inputs[i]->ctx();
      if (save_inputs[i]) {
        input_info.outputs.emplace_back(*inputs[i]);
      } else {
        // Put a dummy array here since it will not be used.
        input_info.outputs.emplace_back();
        input_info.outputs.back().shape_ = inputs[i]->shape();
        input_info.outputs.back().dtype_ = inputs[i]->dtype();
        input_info.outputs.back().storage_type_ = inputs[i]->storage_type();
      }
      inputs[i]->autograd_entry_ = std::move(entry);  // assign last to prevent cyclic reference
    } else if (save_inputs[i]) {
      nnvm::NodeEntry& entry = inputs[i]->autograd_entry_;
      AGInfo::Get(entry.node).outputs[entry.index] = inputs[i]->Detach();
    }
    node->inputs[i] = inputs[i]->autograd_entry_;
  }

  for (auto output : outputs) {
    CHECK(AGInfo::IsNone(*output))
        << "NotImplementedError: Inplace operations (+=, -=, x[:]=, etc) "
        << "are not supported when recording with autograd.";
  }

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    if (save_outputs[i]) {
      info.outputs.emplace_back(outputs[i]->Detach());
    } else {
      // Put a dummy array here since it will not be used.
      info.outputs.emplace_back();
      info.outputs.back().shape_ = outputs[i]->shape();
      info.outputs.back().dtype_ = outputs[i]->dtype();
      info.outputs.back().storage_type_ = outputs[i]->storage_type();
    }
    outputs[i]->autograd_entry_ = nnvm::NodeEntry{node, i, 0};
  }
}

void Imperative::RecordDeferredCompute(nnvm::NodeAttrs &&attrs,
                                       const std::vector<NDArray *> &inputs,
                                       const std::vector<NDArray *> &outputs) {
  CHECK(!is_recording())
      << "MXNetError: Autograd recording is not supported during deferred compute mode.";

  for (const NDArray *input : inputs) {
    CHECK(!DCInfo::IsNone(*input))
        << "ValueError: All inputs to deferred compute recording must be associated "
        << "with a symbolic variable or be the output of a deferred compute operator.";
  }
  for (const NDArray *output : outputs) {
    CHECK(DCInfo::IsNone(*output))
        << "NotImplementedError: Inplace operations (+=, -=, x[:]=, etc) "
        << "are not supported when recording in deferred compute mode.";
  }
  DispatchMode dispatch_mode = DispatchMode::kUndefined;
  Context ctx = imperative::GetContext(attrs, inputs, outputs, Context::CPU());
  imperative::SetShapeType(ctx, attrs, inputs, outputs, &dispatch_mode);

  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->inputs.reserve(inputs.size());
  // Get NodeEntries for inputs
  for (const NDArray *array : inputs) {
    CHECK(array->deferredcompute_entry_.node);  // Must not be nullptr
    node->inputs.emplace_back(array->deferredcompute_entry_);
  }
  node->attrs = std::move(attrs);
  // Need to support NameManager in imperative API to better name node->attrs.name
  node->attrs.name = "node_" + std::to_string(node_count_++);

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    outputs[i]->deferredcompute_entry_ = nnvm::NodeEntry{node, i, 0};
  }

  DCInfo::Create(node, inputs, outputs);
}

nnvm::Symbol Imperative::GetDeferredComputeSymbol(const std::vector<NDArray *> &outputs) {
  nnvm::Symbol s;
  s.outputs.reserve(outputs.size());
  for (NDArray * ndoutput : outputs) {
    CHECK(!Imperative::DCInfo::IsNone(*ndoutput))
        << "ValueError: output_arrays for GetDeferredComputeSymbol "
        << "must have a deferred compute history associated with them.";
    s.outputs.emplace_back(ndoutput->deferredcompute_entry_);
  }
  return s.Copy();
}

void Imperative::SetDeferredComputeVariable(NDArrayHandle *arrays,
                                            SymbolHandle *variables, const int num) {
  // Sanity check all inputs
  for (int i = 0; i < num; i++) {
    nnvm::Symbol *s = reinterpret_cast<nnvm::Symbol *>(variables[i]);
    NDArray *nd = reinterpret_cast<NDArray *>(arrays[i]);
    CHECK_EQ(s->outputs.size(), 1)
        << "MXNDArraySetDeferredComputeVariable expects variables as input. "
        << "Instead got a Symbol with " << s->outputs.size()
        << " outputs as input " << i;
    CHECK(s->outputs[0].node->is_variable())
        << "MXNDArraySetDeferredComputeVariable expects variables as input. "
        << "Instead got a Symbol associated with an operator as input " << i;
    CHECK(DCInfo::IsNone(*nd) || nd->deferredcompute_entry_.node == s->outputs[0].node)
        << "ValueError: array " << i << " is already associated with a different variable. "
        << "You can call array.detach() to obtain a copy without the variable";
  }

  // Store variables in DCInfo of arrays
  for (int i = 0; i < num; i++) {
    nnvm::Symbol *s = reinterpret_cast<nnvm::Symbol *>(variables[i]);
    NDArray *nd = reinterpret_cast<NDArray *>(arrays[i]);
    nd->deferredcompute_entry_ = nnvm::NodeEntry{s->outputs[0].node, 0, 0};

    std::vector<NDArray *> inputs;
    std::vector<NDArray *> outputs;  // No need to specify outputs, as we will set is_computed_
    Imperative::DCInfo& info = Imperative::DCInfo::Create(s->outputs[0].node, inputs, outputs);
    info.is_computed_ = true;
  }
}

std::vector<NDArray*> Imperative::Backward(
    const std::vector<NDArray*>& outputs,
    const std::vector<NDArray*>& ograds,
    const std::vector<NDArray*>& variables,
    bool is_train, bool retain_graph,
    bool create_graph) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const Op* copy_op = Op::Get("_copy");

  // Construct forward graph
  Graph graph;
  graph.outputs.reserve(outputs.size());
  for (const auto& i : outputs) {
    CHECK(!AGInfo::IsNone(*i))
      << "Cannot differentiate node because it is not in a computational graph. "
      << "You need to set is_recording to true or use autograd.record() to save "
      << "computational graphs for backward. If you want to differentiate the same "
      << "graph twice, you need to pass retain_graph=True to backward.";
    graph.outputs.emplace_back(i->autograd_entry_);
  }
  size_t num_forward_outputs = graph.outputs.size();

  // Prepare head gradients
  std::vector<NodeEntry> ograd_entries;
  ograd_entries.reserve(ograds.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    nnvm::ObjectPtr np = Node::Create();
    np->attrs.name = "_head_grad_" + std::to_string(i);
    ograd_entries.emplace_back(NodeEntry{np, 0, 0});
    AGInfo& info = AGInfo::Create(ograd_entries.back().node);
    info.ctx = outputs[i]->ctx();
    if (ograds[i] != nullptr) {
      info.outputs.emplace_back(*ograds[i]);
    } else {
      info.outputs.emplace_back(outputs[i]->shape(), outputs[i]->ctx(),
                                true, outputs[i]->dtype());
      if (info.outputs.back().shape().Size() != 0) {
        info.outputs.back() = static_cast<real_t>(1.0);
      }
    }
  }

  // Get gradient graph
  Symbol sym;
  sym.outputs = graph.outputs;
  std::vector<NodeEntry> xs;
  std::vector<NDArray*> x_grads;
  std::vector<OpReqType> x_reqs;
  if (variables.size()) {
    xs.reserve(variables.size());
    x_grads.reserve(variables.size());
    x_reqs.reserve(variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      CHECK(!AGInfo::IsNone(*variables[i]) &&
            AGInfo::IsVariable(variables[i]->autograd_entry_.node))
          << "Cannot differentiate with respect to the " << i+1 << "-th variable"
          << " because it does not require gradient.";
      xs.emplace_back(variables[i]->autograd_entry_);
      x_grads.push_back(new NDArray());
      x_reqs.push_back(kWriteTo);
    }
  } else {
    std::vector<ObjectPtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    x_grads.reserve(args.size());
    x_reqs.reserve(args.size());
    for (const auto& i : args) {
      AGInfo& info = AGInfo::Get(i);
      if (info.grad_req == kNullOp) continue;
      xs.emplace_back(NodeEntry{i, 0, 0});
      x_grads.push_back(&info.out_grads[0]);
      x_reqs.push_back(info.grad_req);
      info.fresh_out_grad = true;
    }
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";
  }

  Graph g_graph = pass::MXGradient(
      graph, graph.outputs, xs, ograd_entries,
      mxnet::AggregateGradient, nullptr,
      zero_ops, "_copy");
  CHECK_EQ(g_graph.outputs.size(), xs.size());
  for (const auto& e : g_graph.outputs) {
    if (e.node->op() == nullptr) {
      auto node = Node::Create();
      node->attrs.op = copy_op;
      node->inputs.push_back(e);
      graph.outputs.emplace_back(std::move(node));
    } else {
      graph.outputs.push_back(e);
    }
  }
  const auto& idx = graph.indexed_graph();
  // get number of nodes used in forward pass
  size_t num_forward_nodes = 0;
  size_t num_forward_entries = 0;
  for (size_t i = 0; i < num_forward_outputs; ++i) {
    num_forward_nodes = std::max(
        num_forward_nodes, static_cast<size_t>(idx.outputs()[i].node_id + 1));
    num_forward_entries = std::max(
        num_forward_entries, static_cast<size_t>(idx.entry_id(idx.outputs()[i])) + 1);
  }

  // Allocate buffer
  std::vector<NDArray> buff(idx.num_node_entries());
  std::vector<uint32_t> ref_count(buff.size(), 0);
  std::vector<OpStatePtr> states;
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (auto& buffered_array : buff) {
    arrays.push_back(&buffered_array);
  }
  if (create_graph) {
    states.resize(num_forward_nodes);
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& n) {
      AGInfo& info = AGInfo::Get(n);
      states[idx.node_id(n.get())] = info.state;
      for (uint32_t i = 0; i < info.outputs.size(); ++i) {
        CHECK(idx.exist(n.get()));
        size_t nid = idx.node_id(n.get());
        size_t eid = idx.entry_id(nid, i);
        buff[eid] = info.outputs[i];
        buff[eid].autograd_entry_ = NodeEntry{n, i, 0};
        ref_count[eid] = 1;
      }
    });
    for (auto& ograd_entry : ograd_entries) {
      AGInfo& info = AGInfo::Get(ograd_entry.node);
      if (!idx.exist(ograd_entry.node.get())) continue;
      size_t eid = idx.entry_id(ograd_entry);
      buff[eid] = info.outputs[0];
      buff[eid].autograd_entry_ = ograd_entry;
    }
  } else {
    states.reserve(num_forward_nodes);
    for (size_t i = 0; i < num_forward_nodes; ++i) {
      const AGInfo& info = dmlc::get<AGInfo>(idx[i].source->info);
      states.emplace_back(info.state);
      for (size_t j = 0; j < info.outputs.size(); ++j) {
        size_t eid = idx.entry_id(i, j);
        arrays[eid] = const_cast<NDArray*>(&(info.outputs[j]));

        if (retain_graph || info.grad_req != kNullOp) ref_count[eid] = 1;
      }
    }
    for (auto& ograd_entry : ograd_entries) {
      if (!idx.exist(ograd_entry.node.get())) continue;
      AGInfo& info = AGInfo::Get(ograd_entry.node);
      arrays[idx.entry_id(ograd_entry)] = &info.outputs[0];
    }
  }
  for (size_t i = num_forward_outputs; i < graph.outputs.size(); ++i) {
    size_t eid = idx.entry_id(graph.outputs[i]);
    arrays[eid] = x_grads[i - num_forward_outputs];
    ref_count[eid] = 1;
  }

  // Assign context
  auto vctx = PlaceDevice(idx);

  // Infer shape type
  {
    std::pair<uint32_t, uint32_t> node_range, entry_range;
    node_range = {num_forward_nodes, idx.num_nodes()};
    entry_range = {num_forward_entries, idx.num_node_entries()};

    ShapeVector shapes;
    shapes.reserve(idx.num_node_entries());
    bool contain_unknown = false;
    for (const auto& i : arrays) shapes.emplace_back(i->shape());
    CheckAndInferShape(&graph, std::move(shapes), false,
                       node_range, entry_range, &contain_unknown);

    DTypeVector dtypes;
    dtypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) dtypes.emplace_back(i->dtype());
    CheckAndInferType(&graph, std::move(dtypes), false,
                      node_range, entry_range);

    StorageTypeVector stypes;
    stypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) stypes.emplace_back(i->storage_type());
    exec::DevMaskVector dev_mask;
    dev_mask.reserve(idx.num_nodes());
    for (const auto& i : vctx) dev_mask.emplace_back(i.dev_mask());
    CheckAndInferStorageType(&graph, std::move(dev_mask), std::move(stypes), false,
                             node_range, entry_range);
  }

  // Calculate ref count
  for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
    for (const auto& j : idx[i].inputs) {
       ++ref_count[idx.entry_id(j)];
    }
  }

  // Assign reqs
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = num_forward_entries; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }
  for (size_t i = num_forward_outputs; i < idx.outputs().size(); ++i) {
    size_t eid = idx.entry_id(idx.outputs()[i]);
    array_reqs[eid] = x_reqs[i - num_forward_outputs];
  }

  const auto& shapes = graph.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtypes = graph.GetAttr<DTypeVector>("dtype");
  const auto& stypes = graph.GetAttr<StorageTypeVector>("storage_type");
  const auto& dispatch_modes = graph.GetAttr<DispatchModeVector>("dispatch_mode");

  for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
    auto num_outputs = idx[i].source->num_outputs();
    for (size_t j = 0; j < num_outputs; ++j) {
      auto eid = idx.entry_id(i, j);
      if (arrays[eid]->is_none())
        arrays[eid]->ReInit(static_cast<NDArrayStorageType>(stypes[eid]),
                            shapes[eid], vctx[i], dtypes[eid]);
    }
  }

  if (dmlc::GetEnv("MXNET_MEM_PLAN_VERBOSE_LOGGING", false)) {
    common::LogMemoryPlan(graph);
  }

  // Execution

  bool prev_recording = set_is_recording(create_graph);
  bool prev_training = set_is_training(is_train);
  int prev_bulk_size = Engine::Get()->set_bulk_size(backward_bulk_size_);

  try {
    RunGraph(retain_graph, idx, arrays, num_forward_nodes, idx.num_nodes(),
            std::move(array_reqs), std::move(ref_count), &states, dispatch_modes,
            is_recording());
  } catch (const dmlc::Error& e) {
    Engine::Get()->set_bulk_size(prev_bulk_size);
    set_is_recording(prev_recording);
    set_is_training(prev_training);
    throw e;
  }

  Engine::Get()->set_bulk_size(prev_bulk_size);
  set_is_recording(prev_recording);
  set_is_training(prev_training);

  // Clear history
  if (!retain_graph) {
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr& n) {
      AGInfo::Clear(n);
      n->inputs.clear();
    });
  }

  if (variables.size()) {
    return x_grads;
  }
  return {};
}

Imperative::DCInfo::DCInfo(const std::vector<NDArray *> &inputs,
                           const std::vector<NDArray *> &outputs) {
  this->inputs_.reserve(inputs.size());
  this->input_handles_.reserve(inputs.size());
  for (const NDArray *arr : inputs) {
    CHECK(!arr->is_none());
    this->inputs_.push_back(*arr);
    this->input_handles_.push_back(arr);
  }

  this->outputs_.reserve(outputs.size());
  for (const NDArray *arr : outputs) {
    CHECK(!arr->is_none());
    this->outputs_.push_back(*arr);
  }
}

Imperative::DCInfo &
Imperative::DCInfo::Create(const nnvm::ObjectPtr &node,
                           const std::vector<NDArray *> &inputs,
                           const std::vector<NDArray *> &outputs) {
  node->info.construct<DCInfo>(inputs, outputs);
  return Imperative::DCInfo::Get(node);
}

void Imperative::DCInfo::Compute(const NDArray &arr) {
  if (Imperative::DCInfo::IsComputed(arr)) {
    if (!shape_is_known(arr.shape())) {
      // We can't call arr.WaitToRead(); here, as WaitToRead calls Compute
      // leading to an infinite loop.
      Engine::Get()->WaitForVar(arr.ptr_->var);
      if (shape_is_known(arr.ptr_->storage_shape)) {
        arr.SetShapeFromChunk();
      } else {
        CHECK(shape_is_known(arr.shape()));
      }
    }
    return;
  }

  DCInfo &info = Imperative::DCInfo::Get(arr.deferredcompute_entry_.node);
  info.is_computed_ = true;  // We will Invoke at the end of this function.

  // Recursively compute input arrays
  for (const NDArray &input : info.inputs_) {
    Compute(input);
  }

  // Prepare pointers
  std::vector<NDArray *> ndinputs, ndoutputs;
  ndinputs.reserve(info.inputs_.size());
  ndoutputs.reserve(info.outputs_.size());
  for (NDArray &input : info.inputs_)
    ndinputs.push_back(&input);
  for (NDArray &output : info.outputs_)
    ndoutputs.push_back(&output);

  // Compute this array
  Imperative::Get()->Invoke(Context::CPU(),
                            arr.deferredcompute_entry_.node->attrs, ndinputs,
                            ndoutputs);
  if (!shape_is_known(arr.shape())) {
      arr.WaitToRead();
      arr.SetShapeFromChunk();
  }

  // Deallocate copies
  info.inputs_.clear();
  info.outputs_.clear();
}

}  // namespace mxnet
