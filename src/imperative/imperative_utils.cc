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
#include "../operator/operator_common.h"

namespace {

std::vector<NDArray*> NodeInputs(const nnvm::IndexedGraph& idx,
                                 const int node_idx,
                                 const std::vector<NDArray*>& arrays) {
  const nnvm::IndexedGraph::Node& node = idx[node_idx];
  const size_t num_inputs = node.inputs.size();
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (const auto& j : node.inputs) {
    const size_t eid = idx.entry_id(j);
    ndinputs.emplace_back(arrays[eid]);
  }
  return ndinputs;
}

std::vector<NDArray*> NodeOutputs(const nnvm::IndexedGraph& idx,
                                  const int node_idx,
                                  const std::vector<NDArray*>& arrays) {
  const nnvm::IndexedGraph::Node& node = idx[node_idx];
  const size_t num_outputs = node.source->num_outputs();
  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(num_outputs);
  for (size_t j = 0; j < num_outputs; ++j) {
    const size_t eid = idx.entry_id(node_idx, j);
    ndoutputs.emplace_back(arrays[eid]);
  }
  return ndoutputs;
}

std::vector<OpReqType> NodeReq(const nnvm::IndexedGraph& idx,
                               const int node_idx,
                               const std::vector<OpReqType>& array_reqs) {
  const nnvm::IndexedGraph::Node& node = idx[node_idx];
  const size_t num_outputs = node.source->num_outputs();
  std::vector<OpReqType> req;
  req.reserve(num_outputs);
  for (size_t j = 0; j < num_outputs; ++j) {
    const size_t eid = idx.entry_id(node_idx, j);
    req.push_back(array_reqs[eid]);
  }
  return req;
}

void InvokeOperator(const nnvm::IndexedGraph& idx,
                    const int node_idx,
                    const bool retain_graph,
                    const std::vector<NDArray*>& arrays,
                    Context ctx,
                    std::vector<OpStatePtr>* p_states,
                    const std::vector<NDArray*>& ndinputs,
                    const std::vector<NDArray*>& ndoutputs,
                    std::vector<OpReqType> *p_req,
                    std::vector<uint32_t> *p_ref_count,
                    std::function<void(const OpStatePtr &state)> invoke) {
  static const auto bwd_cached_op = Op::Get("_backward_CachedOp");
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");
  std::vector<OpStatePtr>& states = *p_states;
  std::vector<OpReqType> &req = *p_req;
  std::vector<uint32_t> &ref_count = *p_ref_count;

  const nnvm::IndexedGraph::Node& node = idx[node_idx];
  if (node.source->op() == bwd_cached_op) {
    const auto& cached_op = dmlc::get<CachedOpPtr>(node.source->attrs.parsed);
    nnvm::Node* fwd_node = node.source->control_deps[0].get();
    auto fwd_node_id = idx.node_id(fwd_node);
    cached_op->Backward(retain_graph, states[fwd_node_id], ndinputs, req, ndoutputs);
  } else if (createop.count(node.source->op())) {
    mxnet::ShapeVector arg_shapes;
    nnvm::DTypeVector arg_dtypes;
    arg_shapes.reserve(ndinputs.size());
    arg_dtypes.reserve(ndinputs.size());
    for (auto& ndinput : ndinputs) {
      arg_shapes.emplace_back(ndinput->shape());
      arg_dtypes.emplace_back(ndinput->dtype());
    }
    states[node_idx] = createop[node.source->op()](node.source->attrs, ctx, arg_shapes, arg_dtypes);
    invoke(states[node_idx]);
  } else if (is_layer_backward.get(node.source->op(), false)) {
    nnvm::Node* fwd_node = node.source->control_deps[0].get();
    auto fwd_node_id = idx.node_id(fwd_node);
    invoke(states[fwd_node_id]);
  } else {
    invoke(OpStatePtr());
  }
  for (const auto& j : node.inputs) {
    size_t eid = idx.entry_id(j);
    --ref_count[eid];
    if (ref_count[eid] == 0) {
      *arrays[eid] = NDArray();
    }
  }
  for (size_t j = 0; j < ndoutputs.size(); ++j) {
    size_t eid = idx.entry_id(node_idx, j);
    if (ref_count[eid] == 0) {
      *arrays[eid] = NDArray();
    }
  }
}

}  // namespace

namespace mxnet {
namespace imperative {

void RunGraph(
    const bool retain_graph,
    const nnvm::IndexedGraph& idx,
    const std::vector<NDArray*>& arrays,
    size_t node_start, size_t node_end,
    std::vector<OpReqType>&& array_reqs,
    std::vector<uint32_t>&& ref_count,
    std::vector<OpStatePtr> *p_states,
    const DispatchModeVector &dispatch_modes,
    bool recording,
    mxnet::ShapeVector *shapes,
    const imperative::CachedOpMonCallback& callback,
    const bool monitor_all) {
  CHECK(shapes == nullptr);
  for (size_t i = node_start; i < node_end; ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->op() == nullptr) {
      continue;
    }
    std::vector<NDArray*> ndinputs = NodeInputs(idx, i, arrays);
    std::vector<NDArray*> ndoutputs = NodeOutputs(idx, i, arrays);
    std::vector<OpReqType> req = NodeReq(idx, i, array_reqs);
    Context ctx = ndoutputs[0]->ctx();
    if (callback && monitor_all) {
        mxnet::common::ExecuteMonInputCallback(idx, arrays, i, callback);
    }
    auto invoke = [&](const OpStatePtr &state) {
      const nnvm::IndexedGraph::Node& node = idx[i];
      DispatchMode dispatch_mode = dispatch_modes[i];
      Imperative::Get()->InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs,
                                  req, dispatch_mode, state);
      if (recording) {
        Imperative::Get()->RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, state);
      }
    };
    InvokeOperator(idx, i, retain_graph, arrays, ctx, p_states, ndinputs, ndoutputs,
                   &req, &ref_count, invoke);
    if (callback) {
        mxnet::common::ExecuteMonOutputCallback(idx, arrays, i, callback);
    }
  }
}

void NaiveRunGraph(
    const bool retain_graph,
    const Context& default_ctx,
    const nnvm::IndexedGraph& idx,
    const std::vector<NDArray*>& arrays,
    size_t node_start, size_t node_end,
    std::vector<OpReqType>&& array_reqs,
    std::vector<uint32_t>&& ref_count,
    std::vector<OpStatePtr> *p_states,
    const DispatchModeVector &dispatch_modes,
    bool recording,
    mxnet::ShapeVector *shapes,
    const imperative::CachedOpMonCallback& callback,
    const bool monitor_all) {
  for (size_t i = node_start; i < node_end; ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->op() == nullptr) {
      continue;
    }
    std::vector<NDArray*> ndinputs = NodeInputs(idx, i, arrays);
    std::vector<NDArray*> ndoutputs = NodeOutputs(idx, i, arrays);
    std::vector<OpReqType> req;
    Context ctx = GetContext(node.source->attrs, ndinputs, ndoutputs, default_ctx);
    if (callback && monitor_all) {
        mxnet::common::ExecuteMonInputCallback(idx, arrays, i, callback);
    }
    auto invoke = [&](const OpStatePtr &state) {
      const nnvm::IndexedGraph::Node& node = idx[i];
      DispatchMode dispatch_mode = DispatchMode::kUndefined;
      SetShapeType(ctx, node.source->attrs, ndinputs, ndoutputs, &dispatch_mode);
      SetWriteInplaceReq(ndinputs, ndoutputs, &req);
      Imperative::Get()->InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs,
                                  req, dispatch_mode, state);
      for (size_t j = 0; j < ndoutputs.size(); ++j) {
        if (mxnet::op::shape_is_none(ndoutputs[j]->shape())) {
          ndoutputs[j]->WaitToRead();
          ndoutputs[j]->SetShapeFromChunk();
        }
        size_t eid = idx.entry_id(i, j);
        auto shape = ndoutputs[j]->shape();
        (*shapes)[eid] = shape;
      }
      if (recording) {
        Imperative::Get()->RecordOp(NodeAttrs(node.source->attrs), ndinputs, ndoutputs, state);
      }
    };
    InvokeOperator(idx, i, retain_graph, arrays, ctx, p_states, ndinputs, ndoutputs,
                   &req, &ref_count, invoke);
    if (callback) {
        mxnet::common::ExecuteMonOutputCallback(idx, arrays, i, callback);
    }
  }
}

}  // namespace imperative
}  // namespace mxnet
