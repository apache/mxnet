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
#include "./cached_op.h"

namespace mxnet {
#if DMLC_CXX11_THREAD_LOCAL
thread_local bool Imperative::is_train_ = false;
thread_local bool Imperative::is_recording_ = false;
thread_local bool Imperative::is_np_shape_ = false;
#else
MX_THREAD_LOCAL bool Imperative::is_train_ = false;
MX_THREAD_LOCAL bool Imperative::is_recording_ = false;
MX_THREAD_LOCAL bool Imperative::is_np_shape_ = false;
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
  for (size_t i = 0; i < outputs.size(); i++) {
    if (!shape_is_known(outputs[i]->shape())) {
      // the WaitToRead overhead here does not seem to be avoidable
      outputs[i]->WaitToRead();
      outputs[i]->SetShapeFromChunk();
    }
  }
  return ret;
}

void Imperative::MarkVariables(
    const std::vector<NDArray*>& variables,
    const std::vector<mx_uint>& grad_reqs,
    const std::vector<NDArray*>& gradients) {
  for (size_t i = 0; i < variables.size(); ++i) {
    std::string str_c(std::to_string(variable_count_++));
    // Add autograd storage for variables and link to the graph
    variables[i]->autograd_ = nnvm::NodeEntry{
        nnvm::Symbol::CreateVariable("var" + str_c).outputs[0].node, 0, 0};
    AGInfo &info = AGInfo::Create(variables[i]->autograd_.node);
    info.outputs.emplace_back(variables[i]->Detach());
    info.out_grads.emplace_back(gradients[i]->Detach());
    info.grad_req = static_cast<OpReqType>(grad_reqs[i]);
    CHECK(info.grad_req < kOpReqTypeMax) << "gradient update request out of range";
    info.ctx = variables[i]->ctx();
    // Handle gradients themselves
    gradients[i]->autograd_ = nnvm::NodeEntry{
        nnvm::Symbol::CreateVariable("grad" + str_c).outputs[0].node, 0, 0};
    AGInfo &grad_info = AGInfo::Create(gradients[i]->autograd_.node);
    grad_info.outputs.emplace_back(gradients[i]->Detach());
    grad_info.ctx = gradients[i]->ctx();
  }
}

void Imperative::GetBackwardDependency(
    const nnvm::NodePtr& node,
    size_t num_inputs, size_t num_outputs,
    std::vector<bool> *save_inputs,
    std::vector<bool> *save_outputs) {
  static auto& fgradient = nnvm::Op::GetAttr<nnvm::FGradient>("FGradient");
  save_inputs->resize(num_inputs);
  save_outputs->resize(num_outputs);
  std::fill(save_inputs->begin(), save_inputs->end(), false);
  std::fill(save_outputs->begin(), save_outputs->end(), false);

  node->inputs.clear();
  node->inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    node->inputs.emplace_back(nullptr, i, 0);
  }

  if (fgradient.count(node->op())) {
    std::vector<nnvm::NodeEntry> ograd_entries;
    ograd_entries.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      ograd_entries.emplace_back(nullptr, i, 1);
    }
    auto igrad_entries = fgradient[node->op()](node, ograd_entries);
    for (const auto& i : igrad_entries) {
      if (i.node == nullptr && i.version == 0) {
        (*save_inputs)[i.index] = true;
      } else if (i.node == node) {
        (*save_outputs)[i.index] = true;
      }
    }
    DFSVisit(igrad_entries, [&](const nnvm::NodePtr& gnode) {
        if (!gnode || gnode == node) return;
        for (const auto& i : gnode->inputs) {
          if (i.node == nullptr && i.version == 0) {
            (*save_inputs)[i.index] = true;
          } else if (i.node == node) {
            (*save_outputs)[i.index] = true;
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

  for (auto output : outputs) {
    CHECK(AGInfo::IsNone(*output))
      << "Assigning to NDArrays that are already in a computational graph "
      << "will cause undefined behavior when evaluating gradients. "
      << "Please call backward first to clear the graph or do this out side of "
      << "a record section. Also note that you cannot use inplace operations "
      << "like +=, *=, relu(x, out=x), y[idx]=x, etc inside a record section.";
  }

  bool need_grad = false;
  for (const auto& i : inputs) {
    if (AGInfo::IsNone(*i)) continue;
    need_grad = true;
    break;
  }
  if (!need_grad) return;

  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs = std::move(attrs);
  node->attrs.name = "node_" + std::to_string(node_count_++) + "_" + node->attrs.op->name;
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

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (AGInfo::IsNone(*(inputs[i]))) {
      nnvm::NodeEntry entry{nnvm::Symbol::CreateVariable(
          "null" + std::to_string(variable_count_++)).outputs[0].node, 0, 0};
      AGInfo& input_info = AGInfo::Create(entry.node);
      input_info.ctx = inputs[i]->ctx();
      if ((*p_save_inputs)[i]) {
        input_info.outputs.emplace_back(*inputs[i]);
      } else {
        // Put a dummy array here since it will not be used.
        input_info.outputs.emplace_back();
        input_info.outputs.back().shape_ = inputs[i]->shape();
        input_info.outputs.back().dtype_ = inputs[i]->dtype();
        input_info.outputs.back().storage_type_ = inputs[i]->storage_type();
      }
      inputs[i]->autograd_ = std::move(entry);  // assign last to prevent cyclic reference
    } else if ((*p_save_inputs)[i]) {
      AGInfo::Get(inputs[i]->autograd_.node).outputs[inputs[i]->autograd_.index] =
          inputs[i]->Detach();
    }
    node->inputs[i] = inputs[i]->autograd_;
  }

  for (auto output : outputs) {
    CHECK(AGInfo::IsNone(*output))
      << "Inplace operations (+=, -=, x[:]=, etc) are not supported when "
      << "recording with autograd.";
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    if ((*p_save_outputs)[i]) {
      info.outputs.emplace_back(outputs[i]->Detach());
    } else {
      // Put a dummy array here since it will not be used.
      info.outputs.emplace_back();
      info.outputs.back().shape_ = outputs[i]->shape();
      info.outputs.back().dtype_ = outputs[i]->dtype();
      info.outputs.back().storage_type_ = outputs[i]->storage_type();
    }
    outputs[i]->autograd_ = nnvm::NodeEntry{node, static_cast<uint32_t>(i), 0};
  }
}

nnvm::Graph Imperative::CreateGraph(const std::vector<NDArray *> &outputs) {
  nnvm::Graph g;
  std::vector<nnvm::NodeEntry> output_nodes;
  output_nodes.reserve(outputs.size());
  for (const auto &i : outputs) {
    CHECK(!AGInfo::IsNone(*i))
      << "Cannot differentiate node because it is not in a computational graph. "
      << "You need to set is_recording to true or use autograd.record() to save "
      << "computational graphs for backward. If you want to differentiate the same "
      << "graph twice, you need to pass retain_graph=True to backward.";
    g.outputs.emplace_back(i->autograd_);
  }
  return g;
}

std::vector<nnvm::NodeEntry> Imperative::CreateHeadGradientNodes(
    const std::vector<NDArray *> &outputs,
    const std::vector<NDArray *> &ograds) {
  using nnvm::NodeEntry;
  using nnvm::Node;
  std::vector<NodeEntry> ograd_entries;
  ograd_entries.reserve(ograds.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    ograd_entries.emplace_back(Node::Create());
    AGInfo &info = AGInfo::Create(ograd_entries.back().node);
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
  return ograd_entries;
}

struct Imperative::GradientVariableNodes {
  std::vector<nnvm::NodeEntry> variable_nodes;
  std::vector<NDArray*> gradients;
  std::vector<OpReqType> op_req_types;
};

Imperative::GradientVariableNodes Imperative::CreateGradientVariableNodes(
    const std::vector<NDArray *> &variables,
    const std::vector<nnvm::NodeEntry> &outputs) {
  GradientVariableNodes var_nodes;
  if (!variables.empty()) {
    var_nodes.variable_nodes.reserve(variables.size());
    var_nodes.gradients.reserve(variables.size());
    var_nodes.op_req_types.reserve(variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      CHECK(!AGInfo::IsNone(*variables[i]) &&
            AGInfo::IsVariable(variables[i]->autograd_.node))
          << "Cannot differentiate with respect to the " << i+1 << "-th variable"
          << " because it does not require gradient. Did you forget attach_grad()?";
      var_nodes.variable_nodes.emplace_back(variables[i]->autograd_);
      var_nodes.gradients.push_back(new NDArray());
      var_nodes.op_req_types.push_back(kWriteTo);
    }
  } else {
    nnvm::Symbol s;
    s.outputs = outputs;
    std::vector<nnvm::NodePtr> input_ro_nodes = s.ListInputs(Symbol::kReadOnlyArgs);
    var_nodes.variable_nodes.reserve(input_ro_nodes.size());
    var_nodes.gradients.reserve(input_ro_nodes.size());
    var_nodes.op_req_types.reserve(input_ro_nodes.size());
    for (const auto& node : input_ro_nodes) {
      AGInfo& info = AGInfo::Get(node);
      if (info.grad_req != kNullOp) {
        var_nodes.variable_nodes.emplace_back(node);
        var_nodes.gradients.push_back(&info.out_grads[0]);
        var_nodes.op_req_types.push_back(info.grad_req);
        info.fresh_out_grad = true;
      }
    }
    CHECK_GT(var_nodes.variable_nodes.size(), 0)
        << "There are no inputs in computation graph that require gradients.";
  }
  return var_nodes;
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

  Graph graph = CreateGraph(outputs);

  // Prepare head gradient nodes
  std::vector<NodeEntry> ograd_entries = CreateHeadGradientNodes(outputs, ograds);

  // Get variable nodes
  GradientVariableNodes gvars = CreateGradientVariableNodes(variables, graph.outputs);

  // Run backward on the graph
  Graph gradient_graph = pass::MXGradient(
      graph, graph.outputs, gvars.variable_nodes, ograd_entries,
      exec::AggregateGradient, nullptr, nullptr,
      zero_ops, "_copy");

  CHECK_EQ(gradient_graph.outputs.size(), gvars.variable_nodes.size());
  std::vector<nnvm::NodeEntry> forward_outputs = graph.outputs;
  const size_t num_forward_outputs = graph.outputs.size();

  // TODO(larroy): move inside pass::MXGradient
  for (const auto& backward_node : gradient_graph.outputs) {
    if (backward_node.node->is_variable()) {
      auto node = Node::Create();
      node->attrs.op = copy_op;
      node->inputs.push_back(backward_node);
      graph.outputs.emplace_back(std::move(node));
    } else {
      graph.outputs.push_back(backward_node);
    }
  }

  auto& indexed_graph = graph.indexed_graph();
  // get number of nodes used in forward pass
  size_t num_forward_nodes = 0;
  size_t num_forward_entries = 0;
  for (size_t i = 0; i < num_forward_outputs; ++i) {
    num_forward_nodes = std::max(
        num_forward_nodes, static_cast<size_t>(indexed_graph.outputs()[i].node_id + 1));
    num_forward_entries = std::max(
        num_forward_entries, static_cast<size_t>(indexed_graph.entry_id(
            indexed_graph.outputs()[i])) + 1);
  }

  // Allocate buffer
  std::vector<NDArray> buff(indexed_graph.num_node_entries());
  std::vector<uint32_t> ref_count(buff.size(), 0);
  std::vector<OpStatePtr> states;
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (auto& buffered_array : buff)
    arrays.push_back(&buffered_array);

  if (create_graph) {
    states.resize(num_forward_nodes);
    nnvm::DFSVisit(forward_outputs, [&](const nnvm::NodePtr& n) {
      const AGInfo& info = AGInfo::Get(n);
      states.at(indexed_graph.node_id(n.get())) = info.state;
      for (size_t i = 0; i < info.outputs.size(); ++i) {
        CHECK(indexed_graph.exist(n.get()));
        const size_t nid = indexed_graph.node_id(n.get());
        const size_t eid = indexed_graph.entry_id(nid, i);
        buff[eid] = info.outputs[i];
        buff[eid].autograd_ = NodeEntry{n, static_cast<uint32_t>(i), 0};
        ref_count[eid] = 1;
      }
    });
    for (auto& ograd_entry : ograd_entries) {
      const AGInfo& info = AGInfo::Get(ograd_entry.node);
      if (!indexed_graph.exist(ograd_entry.node.get())) continue;
      size_t eid = indexed_graph.entry_id(ograd_entry);
      buff[eid] = info.outputs[0];
      buff[eid].autograd_ = ograd_entry;
    }
  } else {
    states.reserve(num_forward_nodes);
    for (size_t i = 0; i < num_forward_nodes; ++i) {
      // TODO(larroy): This is a code smell 💩
      AGInfo& info = const_cast<AGInfo&>(dmlc::get<AGInfo>(indexed_graph[i].source->info));
      states.emplace_back(info.state);
      for (size_t j = 0; j < info.outputs.size(); ++j) {
        const size_t eid = indexed_graph.entry_id(i, j);
        arrays[eid] = &(info.outputs[j]);
        if (retain_graph || info.grad_req != kNullOp)
          ref_count[eid] = 1;
      }
    }
    for (auto& ograd_entry : ograd_entries) {
      if (!indexed_graph.exist(ograd_entry.node.get())) continue;
      AGInfo& info = AGInfo::Get(ograd_entry.node);
      arrays[indexed_graph.entry_id(ograd_entry)] = &info.outputs[0];
    }
  }
  for (size_t i = num_forward_outputs; i < graph.outputs.size(); ++i) {
    size_t eid = indexed_graph.entry_id(graph.outputs[i]);
    arrays[eid] = gvars.gradients[i - num_forward_outputs];
    ref_count[eid] = 1;
  }

  // Assign context
  auto vctx = PlaceDevice(indexed_graph);

  // Infer shape type
  {
    std::pair<uint32_t, uint32_t> node_range, entry_range;
    node_range = {num_forward_nodes, indexed_graph.num_nodes()};
    entry_range = {num_forward_entries, indexed_graph.num_node_entries()};

    ShapeVector shapes;
    shapes.reserve(indexed_graph.num_node_entries());
    bool contain_unknown = false;
    for (const auto& i : arrays) shapes.emplace_back(i->shape());
    CheckAndInferShape(&graph, std::move(shapes), false,
                       node_range, entry_range, &contain_unknown);

    DTypeVector dtypes;
    dtypes.reserve(indexed_graph.num_node_entries());
    for (const auto& i : arrays) dtypes.emplace_back(i->dtype());
    CheckAndInferType(&graph, std::move(dtypes), false,
                      node_range, entry_range);

    StorageTypeVector stypes;
    stypes.reserve(indexed_graph.num_node_entries());
    for (const auto& i : arrays) stypes.emplace_back(i->storage_type());
    exec::DevMaskVector dev_mask;
    dev_mask.reserve(indexed_graph.num_nodes());
    for (const auto& i : vctx) dev_mask.emplace_back(i.dev_mask());
    CheckAndInferStorageType(&graph, std::move(dev_mask), std::move(stypes), false,
                             node_range, entry_range);
  }

  // Calculate ref count
  for (size_t i = num_forward_nodes; i < indexed_graph.num_nodes(); ++i) {
    for (const auto& j : indexed_graph[i].inputs) {
       ++ref_count[indexed_graph.entry_id(j)];
    }
  }

  // Assign reqs
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = num_forward_entries; i < indexed_graph.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }
  for (size_t i = num_forward_outputs; i < indexed_graph.outputs().size(); ++i) {
    size_t eid = indexed_graph.entry_id(indexed_graph.outputs()[i]);
    array_reqs[eid] = gvars.op_req_types[i - num_forward_outputs];
  }

  const auto& shapes = graph.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtypes = graph.GetAttr<DTypeVector>("dtype");
  const auto& stypes = graph.GetAttr<StorageTypeVector>("storage_type");
  const auto& dispatch_modes = graph.GetAttr<DispatchModeVector>("dispatch_mode");

  for (size_t i = num_forward_nodes; i < indexed_graph.num_nodes(); ++i) {
    auto num_outputs = indexed_graph[i].source->num_outputs();
    for (size_t j = 0; j < num_outputs; ++j) {
      auto eid = indexed_graph.entry_id(i, j);
      if (!arrays[eid]->is_none()) continue;
      if (stypes[eid] == kDefaultStorage) {
        *arrays[eid] = NDArray(shapes[eid], vctx[i], true, dtypes[eid]);
      } else {
        *arrays[eid] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                               shapes[eid], vctx[i], true, dtypes[eid]);
      }
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
    RunGraph(retain_graph, indexed_graph, arrays, num_forward_nodes, indexed_graph.num_nodes(),
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
    nnvm::DFSVisit(forward_outputs, [&](const nnvm::NodePtr& n) {
      AGInfo::Clear(n);
      n->inputs.clear();
    });
  }

  if (variables.size()) {
    return gvars.gradients;
  }
  return {};
}

}  // namespace mxnet
