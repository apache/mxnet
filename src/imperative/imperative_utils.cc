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

#include "./imperative_utils.h"
#include "./cached_op.h"

namespace mxnet {
namespace imperative {
void RunGraph(
    const bool retain_graph,
    const nnvm::IndexedGraph& idx,
    const std::vector<NDArray*> arrays,
    size_t node_start, size_t node_end,
    std::vector<OpReqType>&& array_reqs,
    std::vector<uint32_t>&& ref_count,
    std::vector<OpStatePtr> *p_states,
    const DispatchModeVector &dispatch_modes,
    bool recording) {
  using namespace nnvm;
  using namespace imperative;
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");
  static const auto bwd_cached_op = Op::Get("_backward_CachedOp");

  const auto imp = Imperative::Get();

  std::vector<OpStatePtr>& states = *p_states;

  std::vector<NDArray*> ndinputs, ndoutputs;
  ShapeVector arg_shapes;
  DTypeVector arg_dtypes;
  std::vector<OpReqType> req;

  for (size_t i = node_start; i < node_end; ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->op() == nullptr) continue;
    auto num_outputs = node.source->num_outputs();
    ndinputs.clear();
    ndinputs.reserve(node.inputs.size());
    for (const auto& j : node.inputs) {
      ndinputs.emplace_back(arrays[idx.entry_id(j)]);
      CHECK(!ndinputs.back()->is_none()) << idx[j.node_id].source->attrs.name << " " << j.index;
    }
    ndoutputs.clear();
    ndoutputs.reserve(num_outputs);
    req.clear();
    req.reserve(num_outputs);
    for (size_t j = 0; j < num_outputs; ++j) {
      size_t eid = idx.entry_id(i, j);
      ndoutputs.emplace_back(arrays[eid]);
      req.push_back(array_reqs[eid]);
      CHECK(array_reqs[eid] == kNullOp || !ndoutputs.back()->is_none());
    }
    const Context& ctx = ndoutputs[0]->ctx();
    const DispatchMode dispatch_mode = dispatch_modes[i];
    if (node.source->op() == bwd_cached_op) {
      const auto& cached_op = dmlc::get<CachedOpPtr>(node.source->attrs.parsed);
      nnvm::Node* fwd_node = node.source->control_deps[0].get();
      auto fwd_node_id = idx.node_id(fwd_node);
      cached_op->Backward(retain_graph, states[fwd_node_id], ndinputs, req, ndoutputs);
    } else if (createop.count(node.source->op())) {
      arg_shapes.clear();
      arg_dtypes.clear();
      arg_shapes.reserve(ndinputs.size());
      arg_dtypes.reserve(ndinputs.size());
      for (size_t i = 0; i < ndinputs.size(); ++i) {
        arg_shapes.emplace_back(ndinputs[i]->shape());
        arg_dtypes.emplace_back(ndinputs[i]->dtype());
      }
      states[i] = createop[node.source->op()](
          node.source->attrs, ctx, arg_shapes, arg_dtypes);
      imp->InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs, req, dispatch_mode, states[i]);
      if (recording) {
        imp->RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, states[i]);
      }
    } else if (is_layer_backward.get(node.source->op(), false)) {
      nnvm::Node* fwd_node = node.source->control_deps[0].get();
      auto fwd_node_id = idx.node_id(fwd_node);
      imp->InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs,
               req, dispatch_mode, states[fwd_node_id]);
      if (recording) {
        imp->RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, states[fwd_node_id]);
      }
    } else {
      imp->InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs, req, dispatch_mode);
      if (recording) {
        imp->RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs);
      }
    }

    for (const auto& j : node.inputs) {
      size_t eid = idx.entry_id(j);
      --ref_count[eid];
      if (ref_count[eid] == 0) *arrays[eid] = NDArray();
    }
    for (size_t j = 0; j < ndoutputs.size(); ++j) {
      size_t eid = idx.entry_id(i, j);
      if (ref_count[eid] == 0) *arrays[eid] = NDArray();
    }
  }
}

}  // namespace imperative
}  // namespace mxnet
