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
#include "./naive_cached_op.h"
#include "./exec_pass.h"
#include "../profiler/profiler.h"
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"


namespace mxnet {
OpStatePtr NaiveCachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs,
    const Context& default_ctx) {

  CHECK_EQ(inputs.size(), num_inputs());

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

  OpStatePtr op_state;
  try {
    // Initialize
    bool recording = false;
    op_state = OpStatePtr::Create<DynamicRuntime>();
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

    mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
    imperative::NaiveRunGraph(false, default_ctx, idx, arrays, 0, idx.num_nodes(),
                  std::move(array_reqs), std::move(ref_count), &states,
                  dispatch_modes, false, &shapes, nullptr, false, true);
    {
      auto state_ptr = GetCachedOpState(default_ctx);
      auto& state = state_ptr.get_state<CachedOpState>();
      auto copied_shape = shapes;
      std::lock_guard<std::mutex> lock(state.mutex);
      state.info.fwd_graph.attrs["shape"] = std::make_shared<dmlc::any>(std::move(copied_shape));
    }
    g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  } catch (const dmlc::Error& e) {
    throw e;
  }
  return op_state;
}


}  // namespace mxnet
