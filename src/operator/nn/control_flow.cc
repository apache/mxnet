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

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "../operator_common.h"
#include "../../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

void RunGraph(const nnvm::IndexedGraph& idx,
    const std::vector<NDArray*> arrays,
    size_t node_start, size_t node_end,
    std::vector<OpReqType>&& array_reqs,
    std::vector<uint32_t>&& ref_count,
    std::vector<OpStatePtr> *p_states,
    const DispatchModeVector &dispatch_modes) {
  using namespace nnvm;
  using namespace imperative;
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");

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
      CHECK(!ndinputs.back()->is_none()) << idx[j.node_id].source->attrs.name
          << " " << j.index;
    }
    ndoutputs.clear();
    ndoutputs.reserve(num_outputs);
    req.clear();
    req.reserve(num_outputs);
    for (size_t j = 0; j < num_outputs; ++j) {
      size_t eid = idx.entry_id(i, j);
      ndoutputs.emplace_back(arrays[eid]);
      req.push_back(array_reqs[eid]);
      CHECK(!ndoutputs.back()->is_none());
    }
    const Context& ctx = ndoutputs[0]->ctx();
    const DispatchMode dispatch_mode = dispatch_modes[i];
    if (createop.count(node.source->op())) {
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
      Imperative::InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs, req,
                           dispatch_mode, states[i]);
    } else if (is_layer_backward.get(node.source->op(), false)) {
      nnvm::Node* fwd_node = node.source->control_deps[0].get();
      auto fwd_node_id = idx.node_id(fwd_node);
      Imperative::InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs,
                           req, dispatch_mode, states[fwd_node_id]);
    } else {
      Imperative::InvokeOp(ctx, node.source->attrs, ndinputs, ndoutputs,
                           req, dispatch_mode);
    }
  }
}

static void ExecSubgraph(nnvm::Graph &g, const OpContext& ctx,
                         const std::vector<NDArray>& cinputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& coutputs) {
  using namespace nnvm;
  using namespace imperative;
  const auto& idx = g.indexed_graph();
  size_t num_inputs = idx.input_nodes().size();

  CHECK_EQ(num_inputs, cinputs.size())
      << "The subgraph requires " << num_inputs << " but got " << cinputs.size();

  Context default_ctx = cinputs[0].ctx();
  for (size_t i = 0; i < cinputs.size(); ++i) {
    CHECK_EQ(cinputs[i].ctx(), default_ctx)
        << "The subgraph requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name << " is on " << default_ctx
        << " while " << idx[idx.input_nodes()[i]].source->attrs.name << " is on "
        << cinputs[i].ctx();
  }

  // TODO(zhengda) we might want to buffer them.
  std::vector<NDArray> buff;
  std::vector<OpStatePtr> states;
  std::vector<NDArray> inputs = cinputs;
  std::vector<NDArray> outputs = coutputs;

  // Allocate entries
  states.resize(idx.num_nodes());
  buff.resize(idx.num_node_entries());
  states.reserve(idx.num_nodes());
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  for (size_t i = 0; i < num_inputs; ++i) {
    arrays[idx.entry_id(idx.input_nodes()[i], 0)] = &inputs[i];
  }
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    if (!arrays[eid]->is_none()) outputs[i] = arrays[eid]->Detach();
    arrays[eid] = &outputs[i];
  }

  // Allocate memory for the NDArrays
  std::vector<uint32_t> ref_count = g.GetAttr<std::vector<uint32_t> >(
      ctx.is_train ? "full_ref_count" : "forward_ref_count");

  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  }

  const auto& mem_plan = g.GetAttr<MemoryPlanVector>(
      ctx.is_train ? "full_mem_plan" : "forward_mem_plan");
  AllocateMemory(g, idx, default_ctx, 0, idx.num_node_entries(),
                 mem_plan, arrays, &array_reqs);

  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  RunGraph(idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
      std::move(ref_count), &states, dispatch_modes);
}

static void ForeachComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(attrs.g != nullptr);
  nnvm::Graph &g = *attrs.g;

  // If this is inference, we only need the forward memory plan.
  bool has_mem_plan = !ctx.is_train && g.attrs.count("forward_mem_plan");
  // If this is training, we need the full memory plan.
  has_mem_plan = has_mem_plan || (ctx.is_train && g.attrs.count("full_mem_plan"));
  // If we don't have a memory plan yet, we need to create a memory plan.
  if (!has_mem_plan) {
    const auto& idx = g.indexed_graph();
    nnvm::StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
    for (const auto i : idx.input_nodes())
      storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
    const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
    CHECK_EQ(stypes.size(), storage.size());
    for (size_t i = 0; i < stypes.size(); i++) {
      if (stypes[i] != kDefaultStorage)
        storage[i] = exec::kDynamicStorageID;
    }

    auto mem_plan = imperative::PlanMemory(
        &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >(
          ctx.is_train ? "full_ref_count" : "forward_ref_count"));
    // TODO(zhengda) we need to be careful of changing graph attributes.
    // It's not thread-safe.
    g.attrs[ctx.is_train ? "full_mem_plan" : "forward_mem_plan"]
      = std::make_shared<dmlc::any>(std::move(mem_plan));
  }
  size_t len = inputs[0].shape()[0];
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(inputs[0].shape()[0], outputs[0].shape()[0]);
  std::vector<NDArray> subg_inputs(inputs.size());
  std::vector<NDArray> subg_outputs(outputs.size());
  for (size_t i = 1; i < inputs.size(); i++)
    subg_inputs[i] = inputs[i];
  // Here we iterate over the first dimension of the first input array.
  for (size_t i = 0; i < len; i++) {
    subg_inputs[0] = inputs[0].At(i);
    // For the first iteration, the second argument is the second input array,
    // i.e., the initial state.
    if (i == 0)
      subg_inputs[1] = inputs[1];
    else
      // For the rest of the iterations, the second argument is the output from
      // the previous iteration.
      subg_inputs[1] = subg_outputs[0];
    subg_outputs[0] = outputs[0].At(i);

    ExecSubgraph(g, ctx, subg_inputs, req, subg_outputs);
    // We need to wait for the iteration to complete before executing
    // the next one or return from the loop. In this way, we can reuse
    // the memory in the subgraph.
    for (size_t j = 0; j < subg_outputs.size(); j++)
      subg_outputs[j].WaitToRead();
  }
}

static bool ForeachShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_shape,
                         std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 2U);
  nnvm::ShapeVector shape_inputs = *in_shape;
  // foreach iterates over the first input NDArray over the first dimension.
  shape_inputs[0] = TShape(in_shape->at(0).begin() + 1, in_shape->at(0).end());
  bool ret = shape_assign(&shape_inputs[1], shape_inputs[0]);
  CHECK(ret);
  auto g = attrs.g;
  CHECK(g);
  // TODO(zhengda) This can also be called in the execution engine.
  // We need to make it thread-safe.
  imperative::CheckAndInferShape(g.get(), std::move(shape_inputs), true);
  const auto& shapes = g->GetAttr<nnvm::ShapeVector>("shape");
  CHECK(g->outputs.size() == 1);
  uint32_t eid = g->indexed_graph().entry_id(g->outputs[0]);
  const auto& g_out_shape = shapes[eid];
  const auto& in0 = (*in_shape)[0];
  CHECK_EQ(g_out_shape.ndim() + 1, in0.ndim());
  for (size_t i = 1; i < in0.ndim(); i++)
    CHECK_EQ(in0[i], g_out_shape[i - 1]);
  (*out_shape)[0] = in0;
  return true;
}

static bool ForeachType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_type, std::vector<int> *out_type) {
  nnvm::DTypeVector dtype_inputs = *in_type;
  auto g = attrs.g;
  CHECK(g);
  // TODO(zhengda) This can also be called in the execution engine.
  // We need to make it thread-safe.
  imperative::CheckAndInferType(g.get(), std::move(dtype_inputs), true);
  const auto &dtypes = g->GetAttr<nnvm::DTypeVector>("dtype");
  CHECK(g->outputs.size() == 1);
  uint32_t eid = g->indexed_graph().entry_id(g->outputs[0]);
  (*out_type)[0] = dtypes[eid];
  return true;
}

static bool ForeachStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  auto g = attrs.g;
  CHECK(g);
  const auto& idx = g->indexed_graph();
  CHECK(idx.input_nodes().size() == in_attrs->size());
  exec::DevMaskVector dev_masks(idx.num_nodes(), dev_mask);
  StorageTypeVector storage_type_inputs = *in_attrs;
  imperative::CheckAndInferStorageType(g.get(), std::move(dev_masks),
                                       std::move(storage_type_inputs), true);
  *dispatch_mode = DispatchMode::kFComputeEx;
  const auto& stypes = g->GetAttr<StorageTypeVector>("storage_type");
  auto &outputs = idx.outputs();
  CHECK(outputs.size() == out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = stypes[idx.entry_id(outputs[i])];
  }
  return true;
}

NNVM_REGISTER_OP(Foreach)
.describe(R"code(Foreach)code" ADD_FILELINE)
//.set_attr_parser(ParamParser<ForeachParam>)
.set_attr<FInferStorageType>("FInferStorageType", ForeachStorageType)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"fn", "data1", "data2"};
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return 0;
})
.set_attr<nnvm::FInferShape>("FInferShape", ForeachShape)
.set_attr<nnvm::FInferType>("FInferType", ForeachType)
.set_attr<FComputeEx>("FComputeEx<cpu>", ForeachComputeExCPU)
.add_argument("fn", "Symbol", "Input graph.")
.add_argument("data1", "NDArray-or-Symbol", "Input1.")
.add_argument("data2", "NDArray-or-Symbol", "Input2.");
//.add_arguments(ForeachParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
