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
#include "./subgraph_op_common.h"

namespace mxnet {
namespace op {

struct ForeachParam : public dmlc::Parameter<ForeachParam> {
  int num_args;
  int dim;
  int num_outputs;
  int num_out_data;
  nnvm::Tuple<dim_t> in_state_locs;
  nnvm::Tuple<dim_t> in_data_locs;
  DMLC_DECLARE_PARAMETER(ForeachParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs.");
    DMLC_DECLARE_FIELD(dim).set_default(1)
    .describe("the dimension of the input array to iterate.");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(num_out_data)
    .describe("The number of output data of the subgraph.");
    DMLC_DECLARE_FIELD(in_state_locs)
    .describe("The locations of loop states among the inputs.");
    DMLC_DECLARE_FIELD(in_data_locs)
    .describe("The locations of input data among the inputs.");
  }
};  // struct ForeachParam

DMLC_REGISTER_PARAMETER(ForeachParam);

class ForeachState {
  // These are output arrays from all iterations.
  // They also contain the Op state for each CachedOp.
  std::vector<std::vector<NDArray> > all_outputs;
  std::vector<std::vector<NDArray> > all_inputs;
  std::vector<std::vector<NDArray> > all_gradients;
  std::vector<CachedOpPtr> iter_ops;

 public:
  Symbol subgraph_sym;
  nnvm::Graph subgraph;
  ForeachParam params;

  ForeachState(const Symbol &g, const ForeachParam &params) {
    this->subgraph_sym = g;
    this->subgraph.outputs = g.outputs;
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
  // Get input names.
  const auto& idx = subgraph.indexed_graph();
  std::vector<std::string> arg_names(idx.input_nodes().size());
  for (size_t i = 0; i < idx.input_nodes().size(); ++i)
    arg_names[i] = idx[idx.input_nodes()[i]].source->attrs.name;
  // We don't have parameters for the cached op.
  std::unordered_map<std::string, std::vector<NDArray> > params;
  CachedOpPtr op = std::make_shared<Imperative::CachedOp>(subgraph_sym, kwargs,
                                                          arg_names, params);
  // TODO here we only changed the output arrays in the arguments.
  // Will this be a problem?
  // TODO(zhengda) we need to avoid shape inference and memory plan whenever the op is
  // called. Currently, CachedOp allocates memory each time Forward is called.
  // I need to fix this once the PR for static memory allocation in CachedOp is
  // merged. https://github.com/apache/incubator-mxnet/pull/10817
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

  CHECK_GT(iter_ops.size(), iter_no)
      << "We didn't record the computation for iteration " << iter_no;
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

static void ForeachComputeExCPU(const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  ForeachState &state = state_ptr.get_state<ForeachState>();
  const ForeachParam& params = state.params;
  size_t iter_dim = 0;
  CHECK_EQ(outputs.size(), (size_t) params.num_outputs);
  CHECK_GT(params.in_data_locs.ndim(), 0);
  size_t loc0 = params.in_data_locs[0];
  size_t len = inputs[loc0].shape()[iter_dim];
  for (size_t i = 1; i < params.in_data_locs.ndim(); i++) {
    size_t loc = params.in_data_locs[i];
    CHECK_EQ(inputs[loc].shape()[iter_dim], len);
  }
  for (size_t i = 0; i < (size_t) params.num_out_data; i++)
    CHECK_EQ(len, outputs[i].shape()[iter_dim]);

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

  // Initialize the inputs for the subgraph.
  // In each iteration, we need to update the subgraph inputs for input data
  // and the loop states. This initialization helps to get the read-only
  // arrays in the loop.
  std::vector<NDArray> subg_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    // These are the initial states.
    subg_inputs[i] = inputs[i];
  }

  // Here we iterate over the first dimension of the first input array.
  for (size_t i = 0; i < len; i++) {
    // Initialize outputs for the subgraph.
    std::vector<NDArray> *subg_out_curr = subg_outputs[i % 2];
    std::vector<NDArray> *subg_out_prev = subg_outputs[(i + 1) % 2];
    for (int j = 0; j < params.num_out_data; j++)
      (*subg_out_curr)[j] = outputs[j].At(i);
    // When recording for backward computation, we should make sure 
    // that output arrays are actually different in each iteration.
    if (ctx.need_grad && i < len - 1) {
      for (size_t j = params.num_out_data; j < subg_out_curr->size(); j++)
        (*subg_out_curr)[j] = NDArray(outputs[j].shape(), outputs[j].ctx(),
                                      true, outputs[j].dtype());
    } else if (ctx.need_grad && i == len - 1) {
      // For the last iteration, we need to write data to the output array
      // directly.
      for (size_t j = params.num_out_data; j < subg_out_curr->size(); j++)
        (*subg_out_curr)[j] = outputs[j];
    }

    // Initialize inputs for the subgraph.
    // Get a slice from the input data arrays.
    for (size_t j = 0; j < params.in_data_locs.ndim(); j++) {
      size_t loc = params.in_data_locs[j];
      subg_inputs[loc] = inputs[loc].At(i);
    }
    // For the rest of the iterations, the rest of the arguments are the outputs
    // from the previous iteration.
    if (i > 0) {
      for (size_t j = params.num_out_data; j < subg_out_prev->size(); j++) {
        size_t idx = j - params.num_out_data;
        CHECK_LT(params.in_state_locs[idx], subg_inputs.size());
        subg_inputs[params.in_state_locs[idx]] = (*subg_out_prev)[j];
      }
    }

    state.Forward(subg_inputs, req, *subg_out_curr, ctx.need_grad);
    // We need to wait for the iteration to complete before executing
    // the next one or return from the loop. In this way, we can reuse
    // the memory in the subgraph.
    for (size_t j = 0; j < subg_out_curr->size(); j++) {
      (*subg_out_curr)[j].WaitToRead();
    }
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
  CHECK_GT(params.in_data_locs.ndim(), 0);
  size_t iter_dim = 0;
  std::unordered_set<size_t> in_data_locs(params.in_data_locs.begin(),
                                          params.in_data_locs.end());
  std::unordered_set<size_t> in_state_locs(params.in_state_locs.begin(),
                                           params.in_state_locs.end());
  // The inputs contain out gradients, inputs and outputs.
  int len = inputs[0].shape()[iter_dim];
  size_t num_output_data = params.num_out_data;

  // In backward computation, we need to run iterations from backwards.
  std::vector<NDArray> ograds(params.num_outputs);
  std::vector<NDArray> igrads(outputs.size());
  for (size_t i = num_output_data; i < ograds.size(); i++)
    ograds[i] = inputs[i];
  std::vector<OpReqType> iter_req(req.size());
  for (auto r : req)
    CHECK_NE(r, kWriteInplace);
  for (int iter_num = len - 1; iter_num >= 0; iter_num--) {
    for (int i = 0; i < params.num_out_data; i++)
      ograds[i] = inputs[i].At(iter_num);

    // There are three types of arrays in igrads.
    // * data gradients.
    // * loop variable gradients.
    // * read-only variable gradients.
    // These are the input data gradients.
    for (size_t i = 0; i < igrads.size(); i++) {
      // data gradients.
      if (in_data_locs.count(i)) {
        igrads[i] = outputs[i].At(iter_num);
        iter_req[i] = req[i];
        continue;
      }

      bool in_state = in_state_locs.count(i);
      if (iter_num != 0 && in_state) {
        // For state gradients, we need to allocate new NDArrays
        // because intermediate state gradients won't be returned to the users.
        igrads[i] = NDArray(outputs[i].shape(), outputs[i].ctx(),
                            true, outputs[i].dtype());
      } else {
        igrads[i] = outputs[i];
      }
      if (in_state)
        // For the first iteration, we need to use the request provided by
        // the user to write state gradients to the outputs.
        iter_req[i] = iter_num != 0 ? kWriteTo : req[i];
      else
        // For all read-only variable gradients, we need to use the request
        // provided by the user in the last iteration and later on add gradients
        // to the output arrays.
        iter_req[i] = iter_num == len - 1 ? req[i]: kAddTo;
    }

    state.Backward(iter_num, ograds, iter_req, igrads);

    // We need to wait for the iteration to complete before executing
    // the next one or return from the loop. In this way, we can reuse
    // the memory in the subgraph.
    for (size_t i = 0; i < igrads.size(); i++) {
      igrads[i].WaitToRead();
    }

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
  size_t loc0 = params.in_data_locs[0];
  size_t len = in_shape->at(loc0)[0];
  for (size_t i = 0; i < params.in_data_locs.ndim(); i++) {
    size_t loc = params.in_data_locs[i];
    CHECK_EQ(len, in_shape->at(loc)[0]);
    shape_inputs[loc] = TShape(in_shape->at(loc).begin() + 1, in_shape->at(loc).end());
  }
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  nnvm::Graph g;
  g.outputs = attrs.subgraphs[0]->outputs;
  const auto& idx = g.indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_shape->size());
  CHECK_EQ(idx.outputs().size(), out_shape->size());
  imperative::CheckAndInferShape(&g, std::move(shape_inputs), true);

  const auto& shapes = g.GetAttr<nnvm::ShapeVector>("shape");
  // Inferring the shape in the subgraph may infer the shape of the inputs.
  // We need to copy the inferred input shapes back.
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_shape->size());
  for (size_t i = 0; i < in_shape->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    // If the input shape is none, we should update them.
    if ((*in_shape)[i].ndim() == 0 || (*in_shape)[i].Size() == 0)
      SHAPE_ASSIGN_CHECK(*in_shape, i, shapes[eid]);
  }

  // For the shape of output data.
  for (int i = 0; i < params.num_out_data; i++) {
    uint32_t eid = idx.entry_id(g.outputs[i]);
    const auto& g_out_shape = shapes[eid];
    auto out = TShape(g_out_shape.ndim() + 1);
    out[0] = len;
    for (size_t i = 1; i < out.ndim(); i++)
      out[i] = g_out_shape[i - 1];
    SHAPE_ASSIGN_CHECK(*out_shape, i, out);
  }

  // For the remaining shapes.
  for (size_t i = params.num_out_data; i < g.outputs.size(); i++) {
    uint32_t eid = idx.entry_id(g.outputs[i]);
    SHAPE_ASSIGN_CHECK(*out_shape, i, shapes[eid]);
  }
  size_t num_states = g.outputs.size() - params.num_out_data;
  for (size_t i = 0; i < num_states; i++) {
    size_t loc = params.in_state_locs[i];
    CHECK((*out_shape)[i + params.num_out_data] == (*in_shape)[loc]);
  }
  return true;
}

static bool ForeachType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_type, std::vector<int> *out_type) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  return InferSubgraphDataType(*attrs.subgraphs[0], in_type, out_type);
}

static bool ForeachStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  return InferSubgraphStorage(*attrs.subgraphs[0], dev_mask,
                              dispatch_mode, in_attrs, out_attrs);
}

static bool BackwardForeachStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_args - 1);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  return InferSubgraphBackwardStorage(*attrs.subgraphs[0], dev_mask,
                                      dispatch_mode, in_attrs, out_attrs);
}

static OpStatePtr CreateForeachState(const NodeAttrs& attrs,
                                     Context ctx,
                                     const std::vector<TShape>& ishape,
                                     const std::vector<int>& itype) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  return OpStatePtr::Create<ForeachState>(*attrs.subgraphs[0], params);
}

static std::vector<nnvm::NodeEntry>
ForeachGradient(const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
  ElemwiseGradUseInOut fgrad{"_backward_foreach"};
  std::vector<nnvm::NodeEntry> entries = fgrad(n, ograds);
  entries[0].node->attrs.subgraphs = n->attrs.subgraphs;
  return entries;
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
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  std::vector<std::string> names;
  names.push_back("fn");
  for (int i = 0; i < params.num_args - 1; i++)
    names.push_back("data" + std::to_string(i));
  return names;
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return std::vector<uint32_t>{0};
})
.set_attr<nnvm::FGradient>("FGradient", ForeachGradient)
.set_attr<FCreateOpState>("FCreateOpState", CreateForeachState)
.set_attr<nnvm::FInferShape>("FInferShape", ForeachShape)
.set_attr<nnvm::FInferType>("FInferType", ForeachType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ForeachComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("fn", "Symbol", "Input graph.")
.add_argument("data", "NDArray-or-Symbol[]",
              "The input arrays that include data arrays and states.")
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
.set_attr_parser(ParamParser<ForeachParam>)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ForeachGradComputeExCPU);

}  // namespace op
}  // namespace mxnet
