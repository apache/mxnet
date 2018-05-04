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
#include "../elemwise_op_common.h"
#include "../../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

struct ForeachParam : public dmlc::Parameter<ForeachParam> {
  int num_args;
  int dim;
  int num_outputs;
  nnvm::Tuple<dim_t> in_state_locs;
  DMLC_DECLARE_PARAMETER(ForeachParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs.");
    DMLC_DECLARE_FIELD(dim).set_default(1)
    .describe("the dimension of the input array to iterate.");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(in_state_locs)
    .describe("The locations of loop states among the inputs.");
  }
};  // struct ForeachParam

DMLC_REGISTER_PARAMETER(ForeachParam);

// The input arguments are ordered in the following order:
// in, state0, state1, ...
// We need to reorder them in the same order as the input nodes of the subgraph.
template<typename T>
static std::vector<T> ReorderInputs(const std::vector<T> &in, const nnvm::IndexedGraph& idx) {
  std::vector<T> ret(in.size());
  CHECK_EQ(idx.input_nodes().size(), in.size());
  for (size_t i = 0; i < idx.input_nodes().size(); i++) {
    std::string name = idx[idx.input_nodes()[i]].source->attrs.name;
    if (name == "in") {
      ret[i] = in[0];
    } else {
      auto idx_str = name.substr(5);
      int idx = std::stoi(idx_str);
      ret[i] = in[idx + 1];
    }
  }
  return ret;
}

struct ForeachState {
  Symbol subgraph;
  ForeachParam params;
  // These are output arrays from all iterations.
  // They also contain the Op state for each CachedOp.
  std::vector<std::vector<NDArray> > all_outputs;
  std::vector<std::vector<NDArray> > all_inputs;
  std::vector<std::vector<NDArray> > all_gradients;
  std::vector<CachedOpPtr> iter_ops;

  ForeachState(const Symbol &g, const ForeachParam &params) {
    this->subgraph = g;
    this->params = params;
  }

  void Forward(std::vector<NDArray> cinputs,
               const std::vector<OpReqType>& req,
               std::vector<NDArray> coutputs, bool is_recording);
  void Backward(int iter_no, std::vector<NDArray> ograds,
                const std::vector<OpReqType> &req,
                std::vector<NDArray> igrads);
  void Cleanup() {
    all_outputs.clear();
    all_inputs.clear();
    all_gradients.clear();
    iter_ops.clear();
  }
};

void ForeachState::Forward(std::vector<NDArray> cinputs,
                           const std::vector<OpReqType>& req,
                           std::vector<NDArray> coutputs, bool is_recording) {
  using namespace nnvm;
  using namespace imperative;

  bool orig_is_record;
  if (is_recording)
    orig_is_record = Imperative::Get()->set_is_recording(true);
  else
    orig_is_record = Imperative::Get()->is_recording();

  std::vector<NDArray *> inputs(cinputs.size());
  std::vector<NDArray *> outputs(coutputs.size());
  for (size_t i = 0; i < inputs.size(); i++)
    inputs[i] = &cinputs[i];
  for (size_t i = 0; i < outputs.size(); i++)
    outputs[i] = &coutputs[i];

  if (is_recording) {
    all_inputs.push_back(cinputs);
    std::vector<NDArray> gradients(cinputs.size());
    std::vector<NDArray *> input_ptrs(cinputs.size());
    std::vector<NDArray *> gradient_ptrs(cinputs.size());
    std::vector<mx_uint> grad_reqs(cinputs.size());
    for (size_t i = 0; i < gradients.size(); i++) {
      gradients[i] = NDArray(cinputs[i].shape(), cinputs[i].ctx(),
                             true, cinputs[i].dtype());
      input_ptrs[i] = &cinputs[i];
      gradient_ptrs[i] = &gradients[i];
      grad_reqs[i] = kWriteTo;
    }
    Imperative::Get()->MarkVariables(input_ptrs, grad_reqs, gradient_ptrs);;
  }

  std::vector<std::pair<std::string, std::string> > kwargs;
  kwargs.push_back(std::pair<std::string, std::string>("inline_limit", "0"));
  CachedOpPtr op = std::make_shared<Imperative::CachedOp>(subgraph, kwargs);
  // TODO here we only changed the output arrays in the arguments.
  // Will this be a problem?
  op->Forward(nullptr, inputs, outputs);

  if (is_recording) {
    // TODO does this have right inputs and outputs?
    all_outputs.push_back(coutputs);
    iter_ops.push_back(op);
  }

  Imperative::Get()->set_is_recording(orig_is_record);
}

void ForeachState::Backward(int iter_no, std::vector<NDArray> ograds,
                            const std::vector<OpReqType> &req,
                            std::vector<NDArray> igrads) {
  using namespace nnvm;
  using namespace imperative;

  auto op = iter_ops[iter_no];
  std::vector<NDArray *> inputs;
  std::vector<NDArray *> outputs;
  inputs.reserve(op->num_backward_inputs());
  outputs.reserve(op->num_inputs());
  for (size_t i = 0; i < ograds.size(); i++)
    inputs.push_back(&ograds[i]);

  const std::vector<bool> &save_inputs = op->save_inputs();
  const std::vector<bool> &save_outputs = op->save_outputs();
  CHECK_EQ(save_inputs.size(), all_inputs[iter_no].size());
  CHECK_EQ(op->num_outputs(), all_outputs[iter_no].size());
  for (size_t i = 0; i < all_inputs[iter_no].size(); i++) {
    if (save_inputs[i])
      inputs.push_back(&all_inputs[iter_no][i]);
  }
  for (size_t i = 0; i < all_outputs[iter_no].size(); i++) {
    if (save_outputs[i])
      inputs.push_back(&all_outputs[iter_no][i]);
  }
  CHECK_EQ(inputs.size(), op->num_backward_inputs());
  for (size_t i = 0; i < igrads.size(); i++)
    outputs.push_back(&igrads[i]);
  CHECK_EQ(outputs.size(), op->num_inputs());

  // TODO here we only changed the output arrays in the arguments.
  // Will this be a problem?
  CHECK(!Imperative::AGInfo::IsNone(all_outputs[iter_no][0]));
  const nnvm::NodeEntry &node_entry = all_outputs[iter_no][0].GetAutogradEntry();
  OpStatePtr state = Imperative::AGInfo::Get(node_entry.node).state;
  op->Backward(false, state, inputs, req, outputs);
}

static bool is_recording = true;

static void ForeachComputeExCPU(const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  ForeachState &state = state_ptr.get_state<ForeachState>();
  const ForeachParam& params = state.params;
  CHECK_EQ(outputs.size(), (size_t) params.num_outputs);
  size_t len = inputs[0].shape()[0];
  CHECK_EQ(inputs[0].shape()[0], outputs[0].shape()[0]);

  // Initialize the inputs for the subgraph.
  std::vector<NDArray> subg_inputs(inputs.size());
  for (size_t i = 1; i < inputs.size(); i++) {
    // These are the initial states.
    subg_inputs[i] = inputs[i];
  }

  // Initialize the outputs of the subgraph is a little trickier.
  // The states from the previous iteration are used as the inputs of the next
  // iteration, so I have to maintain two arrays, so the inputs and outputs
  // of the subgraph share the same memory.
  std::vector<NDArray> subg_outputs1(outputs.size());
  std::vector<NDArray> subg_outputs2(outputs.size());
  std::vector<NDArray> *subg_outputs[2]{&subg_outputs1, &subg_outputs2};
  // If the length is an odd number, the last iteration will use the first set
  // of outputs. In this way, we don't need to copy the results from the
  // subgraph to the final outputs of the loop.
  if (len % 2 == 1) {
    for (size_t i = 1; i < subg_outputs1.size(); i++) {
      subg_outputs1[i] = outputs[i];
      subg_outputs2[i] = NDArray(outputs[i].shape(), outputs[i].ctx(), true,
                                 outputs[i].dtype());
    }
  } else {
    // Otherwise, we'll use the second set of outputs.
    for (size_t i = 1; i < subg_outputs1.size(); i++) {
      subg_outputs1[i] = NDArray(outputs[i].shape(), outputs[i].ctx(), true,
                                 outputs[i].dtype());
      subg_outputs2[i] = outputs[i];
    }
  }

  // Here we iterate over the first dimension of the first input array.
  for (size_t i = 0; i < len; i++) {
    std::vector<NDArray> *subg_out_curr = subg_outputs[i % 2];
    std::vector<NDArray> *subg_out_prev = subg_outputs[(i + 1) % 2];
    // TODO it might be possible that the data won't be written to the output
    // array directly.
    (*subg_out_curr)[0] = outputs[0].At(i);
    // When recording for backward computation, we should make sure 
    // that output arrays are actually different in each iteration.
    if (is_recording && i < len - 1) {
      for (size_t j = 1; j < subg_out_curr->size(); j++)
        (*subg_out_curr)[j] = NDArray(outputs[j].shape(), outputs[j].ctx(),
                                      true, outputs[j].dtype());
    } else if (is_recording && i == len - 1) {
      // For the last iteration, we need to write data to the output array
      // directly.
      for (size_t j = 1; j < subg_out_curr->size(); j++)
        (*subg_out_curr)[j] = outputs[j];
    }

    // Get a slice from the first input array.
    // TODO how can we be sure that the first subgraph input is the data input?
    subg_inputs[0] = inputs[0].At(i);
    // For the rest of the iterations, the rest of the arguments are the outputs
    // from the previous iteration.
    if (i > 0) {
      for (size_t j = 1; j < subg_out_prev->size(); j++) {
        CHECK_LT(params.in_state_locs[j - 1], subg_inputs.size());
        subg_inputs[params.in_state_locs[j - 1]] = (*subg_out_prev)[j];
      }
    }

    state.Forward(subg_inputs, req, *subg_out_curr, is_recording);
    // We need to wait for the iteration to complete before executing
    // the next one or return from the loop. In this way, we can reuse
    // the memory in the subgraph.
    for (size_t j = 0; j < subg_out_curr->size(); j++)
      (*subg_out_curr)[j].WaitToRead();
  }
}

static void ForeachGradComputeExCPU(const OpStatePtr& state_ptr,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  ForeachState &state = state_ptr.get_state<ForeachState>();
  const ForeachParam& params = state.params;
  CHECK_EQ(outputs.size(), (size_t) params.num_args - 1);
  // The inputs contain out gradients, inputs and outputs.
  size_t len = inputs[0].shape()[0];
  size_t num_input_data = 1;
  size_t num_output_data = 1;

  // In backward computation, we need to run iterations from backwards.
  std::vector<NDArray> ograds(params.num_outputs);
  std::vector<NDArray> igrads(params.num_args - 1);
  for (size_t i = num_output_data; i < ograds.size(); i++)
    ograds[i] = inputs[i];
  for (int iter_num = len - 1; iter_num >= 0; iter_num--) {
    ograds[0] = inputs[0].At(iter_num);
    igrads[0] = outputs[0].At(iter_num);
    // There are three types of arrays in igrads.
    // * data gradients.
    // * loop variable gradients.
    // * read-only variable gradients.
    if (iter_num != 0) {
      for (size_t i = num_input_data; i < igrads.size(); i++)
        igrads[i] = NDArray(outputs[i].shape(), outputs[i].ctx(),
                            true, outputs[i].dtype());
    } else {
      for (size_t i = num_input_data; i < igrads.size(); i++)
        igrads[i] = outputs[i];
    }

    // TODO is req correct here?
    state.Backward(iter_num, ograds, req, igrads);

    // We need to wait for the iteration to complete before executing
    // the next one or return from the loop. In this way, we can reuse
    // the memory in the subgraph.
    for (size_t i = 0; i < igrads.size(); i++)
      igrads[i].WaitToRead();

    size_t num_states = ograds.size() - num_output_data;
    for (size_t i = 0; i < num_states; i++) {
      size_t loc = params.in_state_locs[i];
      CHECK_LT(loc, igrads.size());
      ograds[i + num_output_data] = igrads[loc];
    }
  }
  state.Cleanup();
}

static bool ForeachShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_shape,
                         std::vector<TShape> *out_shape) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_shape->size(), (size_t) params.num_outputs);
  nnvm::ShapeVector shape_inputs = *in_shape;
  // foreach iterates over the first input NDArray over the first dimension.
  shape_inputs[0] = TShape(in_shape->at(0).begin() + 1, in_shape->at(0).end());
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  auto g = std::make_shared<nnvm::Graph>();
  g->outputs = attrs.subgraphs[0]->outputs;
  // TODO(zhengda) We should avoid creating an index graph so many times.
  const auto& idx = g->indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_shape->size());
  CHECK_EQ(idx.outputs().size(), out_shape->size());
  imperative::CheckAndInferShape(g.get(), std::move(shape_inputs), true);
  const auto& shapes = g->GetAttr<nnvm::ShapeVector>("shape");

  // Inferring the shape in the subgraph may infer the shape of the inputs.
  // We need to copy the inferred input shapes back.
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_shape->size());
  size_t num_input_arrays = 1;
  for (size_t i = num_input_arrays; i < in_shape->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    (*in_shape)[i] = shapes[eid];
  }

  // For the first shape.
  uint32_t eid = idx.entry_id(g->outputs[0]);
  const auto& g_out_shape = shapes[eid];
  const auto &in0 = (*in_shape)[0];
  auto &out0 = (*out_shape)[0];
  CHECK_EQ(g_out_shape.ndim() + 1, in0.ndim());
  out0 = in0;
  for (size_t i = 1; i < out0.ndim(); i++)
    out0[i] = g_out_shape[i - 1];

  // For the remaining shapes.
  for (size_t i = 1; i < g->outputs.size(); i++) {
    uint32_t eid = idx.entry_id(g->outputs[i]);
    (*out_shape)[i] = shapes[eid];
  }
  return true;
}

static bool ForeachType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_type, std::vector<int> *out_type) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), (size_t) params.num_outputs);
  nnvm::DTypeVector dtype_inputs = *in_type;
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  auto g = std::make_shared<nnvm::Graph>();
  g->outputs = attrs.subgraphs[0]->outputs;
  // TODO(zhengda) We should avoid creating an index graph so many times.
  const auto& idx = g->indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_type->size());
  CHECK_EQ(idx.outputs().size(), out_type->size());
  imperative::CheckAndInferType(g.get(), std::move(dtype_inputs), true);

  size_t num_input_arrays = 1;
  const auto &dtypes = g->GetAttr<nnvm::DTypeVector>("dtype");

  // Inferring the data type in the subgraph may infer the data type of the inputs.
  // We need to copy the inferred input data types back.
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_type->size());
  for (size_t i = num_input_arrays; i < in_type->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    (*in_type)[i] = dtypes[eid];
  }

  for (size_t i = 0; i < g->outputs.size(); i++)
    (*out_type)[i] = dtypes[idx.entry_id(g->outputs[i])];
  return true;
}

static bool ForeachStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  auto g = std::make_shared<nnvm::Graph>();
  g->outputs = attrs.subgraphs[0]->outputs;
  // TODO(zhengda) We should avoid creating an index graph so many times.
  const auto& idx = g->indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_attrs->size());
  CHECK_EQ(idx.outputs().size(), out_attrs->size());
  exec::DevMaskVector dev_masks(idx.num_nodes(), dev_mask);
  StorageTypeVector storage_type_inputs = *in_attrs;
  imperative::CheckAndInferStorageType(g.get(), std::move(dev_masks),
                                       std::move(storage_type_inputs), true);

  size_t num_input_arrays = 1;
  const auto& stypes = g->GetAttr<StorageTypeVector>("storage_type");

  // Inferring the storage in the subgraph may infer the storage of the inputs.
  // We need to copy the inferred input storage back.
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_attrs->size());
  for (size_t i = num_input_arrays; i < in_attrs->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    (*in_attrs)[i] = stypes[eid];
  }

  *dispatch_mode = DispatchMode::kFComputeEx;
  auto &outputs = idx.outputs();
  CHECK(outputs.size() == out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); i++)
    (*out_attrs)[i] = stypes[idx.entry_id(outputs[i])];
  return true;
}

static bool BackwardForeachStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  // TODO I need to set storage type properly.
  return storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                             dispatch_mode, DispatchMode::kFComputeEx);
}

static OpStatePtr CreateForeachState(const NodeAttrs& attrs,
                                     Context ctx,
                                     const std::vector<TShape>& ishape,
                                     const std::vector<int>& itype) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  return OpStatePtr::Create<ForeachState>(*attrs.subgraphs[0], params);
}

void ForeachParamParser(nnvm::NodeAttrs* attrs) {
  ParamParser<ForeachParam>(attrs);
  // This is to indicate that the operator has a subgraph.
  attrs->subgraphs.resize(1);
}

NNVM_REGISTER_OP(_foreach)
.describe(R"code(foreach)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ForeachParam>)
.set_attr<FInferStorageType>("FInferStorageType", ForeachStorageType)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  return params.num_outputs;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"fn", "data1", "data2"};
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return std::vector<uint32_t>{0};
})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_foreach"})
.set_attr<FCreateOpState>("FCreateOpState", CreateForeachState)
.set_attr<nnvm::FInferShape>("FInferShape", ForeachShape)
.set_attr<nnvm::FInferType>("FInferType", ForeachType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ForeachComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("fn", "Symbol", "Input graph.")
.add_argument("input", "NDArray-or-Symbol", "The input array where we iterate over.")
.add_argument("states", "NDArray-or-Symbol[]", "The list of initial states.")
.add_arguments(ForeachParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_foreach)
.set_num_inputs([](const NodeAttrs& attrs){
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  return params.num_outputs * 2 + params.num_args - 1;
  })
.set_num_outputs([](const NodeAttrs& attrs){
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  return params.num_args - 1;
  })
.set_attr<FInferStorageType>("FInferStorageType", BackwardForeachStorageType)
.set_attr_parser(ForeachParamParser)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ForeachGradComputeExCPU);

}  // namespace op
}  // namespace mxnet
