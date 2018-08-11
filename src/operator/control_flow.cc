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
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "../imperative/imperative_utils.h"
#include "./subgraph_op_common.h"

namespace mxnet {
namespace op {

struct ForeachParam : public dmlc::Parameter<ForeachParam> {
  int num_args;
  int num_outputs;
  int num_out_data;
  // The location of states in the subgraph inputs.
  nnvm::Tuple<dim_t> in_state_locs;
  // The location of data arrays in the subgraph inputs.
  nnvm::Tuple<dim_t> in_data_locs;
  // The location of remaining arrays in the subgraph inputs.
  nnvm::Tuple<dim_t> remain_locs;
  DMLC_DECLARE_PARAMETER(ForeachParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs.");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(num_out_data)
    .describe("The number of output data of the subgraph.");
    DMLC_DECLARE_FIELD(in_state_locs)
    .describe("The locations of loop states among the inputs.");
    DMLC_DECLARE_FIELD(in_data_locs)
    .describe("The locations of input data among the inputs.");
    DMLC_DECLARE_FIELD(remain_locs)
    .describe("The locations of remaining data among the inputs.");
  }
};  // struct ForeachParam

DMLC_REGISTER_PARAMETER(ForeachParam);

class ForeachState: public LoopState {
 public:
  ForeachParam params;
  int num_iterations;

  ForeachState(const Symbol &g, const ForeachParam &params) : LoopState(g) {
    this->params = params;
  }
};

static void ForeachComputeExCPU(const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  ForeachState &state = state_ptr.get_state<ForeachState>();
  const ForeachParam& params = state.params;
  const size_t iter_dim = 0;
  CHECK_EQ(outputs.size(), (size_t) params.num_outputs);
  CHECK_GT(params.in_data_locs.ndim(), 0);
  size_t len = inputs[0].shape()[iter_dim];
  state.num_iterations = len;
  for (size_t i = 1; i < params.in_data_locs.ndim(); i++)
    CHECK_EQ(inputs[i].shape()[iter_dim], len);
  for (size_t i = 0; i < (size_t) params.num_out_data; i++)
    CHECK_EQ(len, outputs[i].shape()[iter_dim]);
  for (const auto &arr : outputs)
    CHECK_EQ(arr.storage_type(), kDefaultStorage)
        << "The for operator doesn't support the sparse format";

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
    for (size_t i = params.num_out_data; i < subg_outputs1.size(); i++) {
      subg_outputs1[i] = outputs[i];
      subg_outputs2[i] = NDArray(outputs[i].shape(), outputs[i].ctx(), true,
                                 outputs[i].dtype());
    }
  } else {
    // Otherwise, we'll use the second set of outputs.
    for (size_t i = params.num_out_data; i < subg_outputs1.size(); i++) {
      subg_outputs1[i] = NDArray(outputs[i].shape(), outputs[i].ctx(), true,
                                 outputs[i].dtype());
      subg_outputs2[i] = outputs[i];
    }
  }

  // Initialize the inputs for the subgraph.
  // In each iteration, we need to update the subgraph inputs for input data
  // and the loop states.
  std::vector<NDArray> subg_inputs(inputs.size());
  // The remaining arrays (other than input data and states) only need to be set once.
  for (size_t j = 0; j < params.remain_locs.ndim(); j++) {
    CHECK_LT(params.remain_locs[j], subg_inputs.size());
    subg_inputs[params.remain_locs[j]] = inputs[j + params.in_data_locs.ndim()
        + params.in_state_locs.ndim()];
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
      subg_inputs[loc] = inputs[j].At(i);
    }
    // For the rest of the iterations, the states are the outputs
    // from the previous iteration.
    if (i > 0) {
      for (size_t j = params.num_out_data; j < subg_out_prev->size(); j++) {
        size_t idx = j - params.num_out_data;
        CHECK_LT(params.in_state_locs[idx], subg_inputs.size());
        subg_inputs[params.in_state_locs[idx]] = (*subg_out_prev)[j];
      }
    } else {
      for (size_t j = 0; j < params.in_state_locs.ndim(); j++) {
        CHECK_LT(params.in_state_locs[j], subg_inputs.size());
        subg_inputs[params.in_state_locs[j]] = inputs[j + params.in_data_locs.ndim()];
      }
    }

    state.Forward(i, subg_inputs, req, *subg_out_curr, ctx.need_grad);
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
  for (const auto &arr : outputs)
    CHECK_EQ(arr.storage_type(), kDefaultStorage)
        << "The for operator doesn't support the sparse format";
  int len = state.num_iterations;
  size_t num_output_data = params.num_out_data;

  // In backward computation, we need to run iterations from backwards.
  std::vector<NDArray> subg_ograds(params.num_outputs);
  std::vector<NDArray> subg_igrads(outputs.size());
  for (size_t i = num_output_data; i < subg_ograds.size(); i++)
    subg_ograds[i] = inputs[i];
  std::vector<OpReqType> subg_req(req.size());
  for (auto r : req)
    CHECK_NE(r, kWriteInplace);

  // There are three types of arrays in igrads.
  // * data gradients.
  // * loop variable gradients.
  // * remaining variable gradients.
  // They are in the following order:
  // [data vars], [loop vars], [remaining vars]

  // [remaining vars]
  for (size_t i = 0; i < params.remain_locs.ndim(); i++) {
    size_t loc = params.remain_locs[i];
    size_t orig_loc = i + params.in_data_locs.ndim() + params.in_state_locs.ndim();
    subg_igrads[loc] = outputs[orig_loc];
    subg_req[loc] = req[orig_loc];
  }

  for (int iter_num = len - 1; iter_num >= 0; iter_num--) {
    for (int i = 0; i < params.num_out_data; i++)
      subg_ograds[i] = inputs[i].At(iter_num);
    if (iter_num < len - 1) {
      // For the rest of the iterations, we should add graidents to the
      // remaining vars.
      for (size_t i = 0; i < params.remain_locs.ndim(); i++) {
        size_t loc = params.remain_locs[i];
        subg_req[loc] = kAddTo;
      }
    }

    // [data vars]
    for (size_t i = 0; i < params.in_data_locs.ndim(); i++) {
      size_t loc = params.in_data_locs[i];
      subg_igrads[loc] = outputs[i].At(iter_num);
      subg_req[loc] = req[i];
    }
    // [loop vars]
    for (size_t i = 0; i < params.in_state_locs.ndim(); i++) {
      size_t loc = params.in_state_locs[i];
      const NDArray &output = outputs[i + params.in_data_locs.ndim()];
      if (iter_num != 0) {
        // For state gradients, we need to allocate new NDArrays
        // because intermediate state gradients won't be returned to the users.
        subg_igrads[loc] = NDArray(output.shape(), output.ctx(), true, output.dtype());
      } else {
        subg_igrads[loc] = output;
      }
      // For the first iteration, we need to use the request provided by
      // the user to write state gradients to the outputs.
      subg_req[loc] = iter_num != 0 ? kWriteTo : req[i + params.in_data_locs.ndim()];
    }

    state.Backward(iter_num, subg_ograds, subg_req, subg_igrads);

    size_t num_states = subg_ograds.size() - num_output_data;
    for (size_t i = 0; i < num_states; i++) {
      size_t loc = params.in_state_locs[i];
      CHECK_LT(loc, subg_igrads.size());
      subg_ograds[i + num_output_data] = subg_igrads[loc];
    }
  }
  state.Cleanup();
}

template<typename T>
static void remap(const std::vector<T> &op_in, size_t start,
                  const nnvm::Tuple<dim_t> &locs, std::vector<T> *subg_in) {
  auto op_in_it = op_in.begin() + start;
  for (size_t i = 0; i < locs.ndim(); i++) {
    dim_t loc = locs[i];
    subg_in->at(loc) = *(op_in_it + i);
  }
}

static inline TShape SliceFirstDim(const TShape &s) {
  if (s.ndim() > 1) {
    return TShape(s.begin() + 1, s.end());
  } else {
    return TShape(mshadow::Shape1(1));
  }
}

static bool ForeachShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_shape,
                         std::vector<TShape> *out_shape) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_shape->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 1U);

  std::vector<TShape> subg_in_shape(in_shape->size());
  // data shape
  std::vector<bool> data_1d(params.in_data_locs.ndim(), false);
  for (size_t i = 0; i < params.in_data_locs.ndim(); i++) {
    size_t loc = params.in_data_locs[i];
    if (in_shape->at(i).ndim() == 1)
      data_1d[i] = true;
    subg_in_shape[loc] = SliceFirstDim(in_shape->at(i));
  }
  // state shape
  remap(*in_shape, params.in_data_locs.ndim(), params.in_state_locs,
        &subg_in_shape);
  // remaining shape
  remap(*in_shape, params.in_data_locs.ndim() + params.in_state_locs.ndim(),
        params.remain_locs, &subg_in_shape);

  std::vector<TShape> subg_out_shape = *out_shape;
  for (int i = 0; i < params.num_out_data; i++) {
    TShape shape = subg_out_shape[i];
    // If we don't have shape info, we don't need to do anything.
    if (shape.ndim() == 0)
      continue;
    subg_out_shape[i] = SliceFirstDim(shape);
  }

  bool infer_success = InferSubgraphShape(*attrs.subgraphs[0],
                                          &subg_in_shape, &subg_out_shape);

  // After inference, we need to move inferred information back to in_shape and
  // out_shape.

  // For the shape of output data.
  size_t len = in_shape->at(0)[0];
  CHECK_GT(len, 0);
  for (int i = 0; i < params.num_out_data; i++) {
    // If the output shape isn't inferred, we don't need to propogate the info.
    const auto& g_out_shape = subg_out_shape[i];
    if (g_out_shape.ndim() == 0)
      continue;

    auto out = TShape(g_out_shape.ndim() + 1);
    out[0] = len;
    for (size_t i = 1; i < out.ndim(); i++)
      out[i] = g_out_shape[i - 1];
    SHAPE_ASSIGN_CHECK(*out_shape, i, out);
  }
  // For the shape of output states.
  for (size_t i = params.num_out_data; i < subg_out_shape.size(); i++)
    SHAPE_ASSIGN_CHECK(*out_shape, i, subg_out_shape[i]);

  // For the shape of input data.
  for (size_t i = 0; i < params.in_data_locs.ndim(); i++) {
    size_t loc = params.in_data_locs[i];
    const auto &shape = subg_in_shape[loc];
    // If the input data shape isn't inferred, we don't need to propogate the
    // info.
    if (shape.ndim() == 0)
      continue;

    if (data_1d[i]) {
      TShape s(1);
      s[0] = len;
      SHAPE_ASSIGN_CHECK(*in_shape, i, s);
    } else {
      auto in = TShape(shape.ndim() + 1);
      in[0] = len;
      for (size_t i = 1; i < in.ndim(); i++)
        in[i] = shape[i - 1];
      SHAPE_ASSIGN_CHECK(*in_shape, i, in);
    }
  }
  // For the shape of state.
  for (size_t i = 0; i < params.in_state_locs.ndim(); i++) {
    size_t loc = params.in_state_locs[i];
    SHAPE_ASSIGN_CHECK(*in_shape, i + params.in_data_locs.ndim(),
                       subg_in_shape[loc]);
  }
  // For the shape of remaining data.
  for (size_t i = 0; i < params.remain_locs.ndim(); i++) {
    size_t loc = params.remain_locs[i];
    SHAPE_ASSIGN_CHECK(*in_shape,
                       i + params.in_data_locs.ndim() + params.in_state_locs.ndim(),
                       subg_in_shape[loc]);
  }

  if (infer_success) {
    size_t num_states = out_shape->size() - params.num_out_data;
    for (size_t i = 0; i < num_states; i++) {
      CHECK_EQ((*out_shape)[i + params.num_out_data],
               (*in_shape)[i + params.in_data_locs.ndim()]);
    }
  }
  // Check if we have inferred the shapes correctly.
  return infer_success;
}

static bool ForeachType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_type, std::vector<int> *out_type) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  std::vector<int> subg_in_type(in_type->size(), 0);
  remap(*in_type, 0, params.in_data_locs, &subg_in_type);
  remap(*in_type, params.in_data_locs.ndim(), params.in_state_locs, &subg_in_type);
  remap(*in_type, params.in_data_locs.ndim() + params.in_state_locs.ndim(),
        params.remain_locs, &subg_in_type);
  bool success = InferSubgraphDataType(*attrs.subgraphs[0], &subg_in_type, out_type);
  for (size_t i = 0; i < params.in_data_locs.ndim(); i++) {
    size_t loc = params.in_data_locs[i];
    TYPE_ASSIGN_CHECK(*in_type, i, subg_in_type[loc]);
  }
  for (size_t i = 0; i < params.in_state_locs.ndim(); i++) {
    size_t loc = params.in_state_locs[i];
    TYPE_ASSIGN_CHECK(*in_type, i + params.in_data_locs.ndim(), subg_in_type[loc]);
  }
  for (size_t i = 0; i < params.remain_locs.ndim(); i++) {
    size_t loc = params.remain_locs[i];
    TYPE_ASSIGN_CHECK(*in_type, i + params.in_data_locs.ndim() + params.in_state_locs.ndim(),
                      subg_in_type[loc]);
  }
  return success;
}

static bool ForeachStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  std::vector<int> subg_in_attrs(in_attrs->size(), kUndefinedStorage);
  remap(*in_attrs, 0, params.in_data_locs, &subg_in_attrs);
  remap(*in_attrs, params.in_data_locs.ndim(), params.in_state_locs, &subg_in_attrs);
  remap(*in_attrs, params.in_data_locs.ndim() + params.in_state_locs.ndim(),
        params.remain_locs, &subg_in_attrs);
  bool success = InferSubgraphStorage(*attrs.subgraphs[0], dev_mask,
                                      dispatch_mode, &subg_in_attrs, out_attrs);
  for (size_t i = 0; i < params.in_data_locs.ndim(); i++) {
    size_t loc = params.in_data_locs[i];
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, i, subg_in_attrs[loc]);
  }
  for (size_t i = 0; i < params.in_state_locs.ndim(); i++) {
    size_t loc = params.in_state_locs[i];
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, i + params.in_data_locs.ndim(),
                              subg_in_attrs[loc]);
  }
  for (size_t i = 0; i < params.remain_locs.ndim(); i++) {
    size_t loc = params.remain_locs[i];
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs,
                              i + params.in_data_locs.ndim() + params.in_state_locs.ndim(),
                              subg_in_attrs[loc]);
  }
  return success;
}

static bool BackwardForeachStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  const ForeachParam& params = nnvm::get<ForeachParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_args - 1);
  CHECK_EQ(in_attrs->size(), (size_t) params.num_args - 1 + params.num_outputs * 2);
  CHECK_EQ(attrs.subgraphs.size(), 1U);
  CachedOp op(*attrs.subgraphs[0],
              std::vector<std::pair<std::string, std::string> >());
  // map the operator inputs to the subgraph inputs.
  std::vector<int> subg_forward_ins(params.num_args - 1, kUndefinedStorage);
  remap(*in_attrs, params.num_outputs, params.in_data_locs, &subg_forward_ins);
  remap(*in_attrs, params.num_outputs + params.in_data_locs.ndim(),
        params.in_state_locs, &subg_forward_ins);
  remap(*in_attrs, params.num_outputs + params.in_data_locs.ndim() + params.in_state_locs.ndim(),
        params.remain_locs, &subg_forward_ins);

  // Copy backward input storage to backward subgraph input storage.
  std::vector<int> subg_in_attrs = *in_attrs;
  for (size_t i = 0; i < subg_forward_ins.size(); i++)
    subg_in_attrs[i + params.num_outputs] = subg_forward_ins[i];
  return op.BackwardStorageType(attrs, dev_mask, dispatch_mode,
                                &subg_in_attrs, out_attrs);
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

struct WhileLoopParam : public dmlc::Parameter<WhileLoopParam> {
  int num_args;
  int num_outputs;
  int num_out_data;
  int max_iterations;
  // `cond' and `func' each takes a subset of while_loop's inputs as that to their subgraphs
  // `cond_input_locs' contains indices of inputs fed to `cond', and
  // `func_input_locs' contains indices of inputs fed to `func'.
  // `func_var_locs' are indices in which input "variables" are stored in func's inputs.
  nnvm::Tuple<dim_t> cond_input_locs;
  nnvm::Tuple<dim_t> func_input_locs;
  nnvm::Tuple<dim_t> func_var_locs;
  DMLC_DECLARE_PARAMETER(WhileLoopParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(2)
    .describe("Number of input arguments, including cond and func as two symbol inputs.");
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(num_out_data).set_lower_bound(0)
    .describe("The number of outputs from the function body.");
    DMLC_DECLARE_FIELD(max_iterations).set_lower_bound(1)
    .describe("Maximum number of iterations.");
    DMLC_DECLARE_FIELD(cond_input_locs)
    .describe("The locations of cond's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(func_input_locs)
    .describe("The locations of func's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(func_var_locs)
    .describe("The locations of loop_vars among func's inputs.");
  }
  template <typename T>
  bool sync_in_out(std::vector<T> *in,
                   std::vector<T> *out,
                   std::function<bool(const T &)> is_empty) const {
    for (int i = this->num_out_data; i < this->num_outputs; ++i) {
      // each out->at(i) is a params, loop_var
      T &x = in->at(this->func_input_locs[this->func_var_locs[i - this->num_out_data]]);
      T &y = out->at(i);
      fill_value(&x, &y, is_empty(x), is_empty(y));
    }
    return true;
  }
};  // struct WhileLoopParam

DMLC_REGISTER_PARAMETER(WhileLoopParam);

class WhileLoopState: public LoopState {
 public:
  WhileLoopParam params;
  size_t n_iterations;  // the actual number of steps taken in this while loop, <= max_iterations
  CachedOpPtr cond_op;
  // abbrev for output_input_mapping
  // indicates to which index the output of `func' will be copied to the input of `cond'
  std::vector<int> oi_map;

  WhileLoopState(const WhileLoopParam &params, const Symbol &cond, const Symbol &func) :
                 LoopState(func),
                 params(params),
                 n_iterations(0U),
                 cond_op(LoopState::MakeSharedOp(cond)),
                 oi_map(params.func_var_locs.ndim(), -1) {
    const nnvm::Tuple<dim_t> &func_input_locs = params.func_input_locs;
    const nnvm::Tuple<dim_t> &func_var_locs = params.func_var_locs;
    const nnvm::Tuple<dim_t> &cond_input_locs = params.cond_input_locs;
    for (size_t i = 0; i < func_var_locs.ndim(); ++i) {
      dim_t pos_i = func_input_locs[func_var_locs[i]];
      for (size_t j = 0; j < cond_input_locs.ndim(); ++j) {
        dim_t pos_j = cond_input_locs[j];
        if (pos_i == pos_j) {
          this->oi_map[i] = j;
        }
      }
    }
  }
};

static void WhileLoopComputeExCPU(const OpStatePtr& state_ptr,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  // The argument `inputs' are loop_vars and other inputs
  // loop_vars are stored in stored in `loop_vars_locs'
  // The argument `outputs' are output and new_loop_vars
  // [0: num_out_data) are outputs at each step.
  // [num_out_data: ) are new_loop_vars
  // TODO(Junru): avoid dynamic NDArray allocation
  WhileLoopState &state = state_ptr.get_state<WhileLoopState>();
  const WhileLoopParam& params = state.params;
  // a helper function, converting std::vector<NDArray> to std::vector<NDArray*>
  const auto to_ptr_vec = [](std::vector<NDArray> &in, std::vector<NDArray*> *out) {
    out->clear();
    out->reserve(in.size());
    std::transform(std::begin(in),
                   std::end(in),
                   std::back_inserter(*out),
                   [](NDArray &a) {return &a;});
  };
  // sanity checks
  CHECK_EQ(inputs.size() + 2U, (size_t) params.num_args);
  CHECK_EQ(outputs.size(), (size_t) params.num_outputs);
  CHECK_EQ(outputs.size(), req.size());
  for (size_t i = 0; i < (size_t) params.num_out_data; i++)
    CHECK_EQ(params.max_iterations, outputs[i].shape()[0]);
  // construct inputs and outputs for cond
  std::vector<NDArray> cond_inputs, cond_outputs = {NDArray()};
  extract_by_loc(inputs, params.cond_input_locs, &cond_inputs);
  std::vector<NDArray*> cond_input_ptr, cond_output_ptr;
  to_ptr_vec(cond_inputs, &cond_input_ptr);
  to_ptr_vec(cond_outputs, &cond_output_ptr);
  // construct inputs and outputs for func
  std::vector<NDArray> func_inputs, func_outputs(outputs.size());
  extract_by_loc(inputs, params.func_input_locs, &func_inputs);
  for (size_t &step = state.n_iterations = 0; step < (size_t) params.max_iterations; ++step) {
    state.cond_op->Forward(nullptr, cond_input_ptr, cond_output_ptr);
    if (!as_bool_scalar(*cond_output_ptr[0])) {
      break;
    }
    // we create func_outputs for the current step:
    // func_outputs[0: num_out_data] is a slice of outputs[][step]
    for (size_t i = 0; i < (size_t) params.num_out_data; ++i) {
      func_outputs[i] = outputs[i].At(step);
    }
    // func_outputs[num_out_data: ] are new_loop_vars, need to allocate new memory
    for (size_t i = params.num_out_data; i < outputs.size(); ++i) {
      func_outputs[i] = NDArray(outputs[i].shape(), outputs[i].ctx(), true, outputs[i].dtype());
    }
    state.Forward(step, func_inputs, req, func_outputs, ctx.need_grad);
    // func_inputs on the next step:
    // the output (new_loop_vars) will become the new inputs (loop_vars)
    for (size_t i = params.num_out_data; i < outputs.size(); ++i) {
      size_t j = params.func_var_locs[i - params.num_out_data];
      CHECK_EQ(func_inputs[j].shape(), func_outputs[i].shape());
      func_inputs[j] = func_outputs[i];
      int k = state.oi_map[i - params.num_out_data];
      if (k != -1) {
        // I actually don't need to update cond_inputs
        cond_inputs[k] = func_outputs[i];
        cond_input_ptr[k] = &func_outputs[i];
      }
    }
  }
  // copy output data to `outputs'
  // case 1: at least one step is executed,
  // the final_loop_vars must be stored in func_inputs
  // case 2: no step is executed
  // the final_loop_vars is the same as loop_vars, which are also stored in func_inputs
  // therefore, we copy func_inputs[:] to outputs[num_out_data: ]
  for (size_t i = params.num_out_data; i < outputs.size(); ++i) {
    size_t j = params.func_var_locs[i - params.num_out_data];
    mxnet::CopyFromTo(func_inputs[j], &outputs[i]);
  }
}

static void WhileLoopGradComputeExCPU(const OpStatePtr& state_ptr,
                                      const OpContext& ctx,
                                      const std::vector<NDArray>& inputs,
                                      const std::vector<OpReqType>& _req,
                                      const std::vector<NDArray>& _outputs) {
  // inputs are dl / df(x)
  // outputs are dl / dx
  // where f is the current function,
  // x is the input to the current function,
  // TODO(Junru): avoid dynamic NDArray allocation
  WhileLoopState &state = state_ptr.get_state<WhileLoopState>();
  const WhileLoopParam& params = state.params;
  // sanity checks
  CHECK_EQ(_outputs.size() + 2U, (size_t) params.num_args);
  CHECK_EQ(_outputs.size(), _req.size());
  for (auto x : _req) {
    CHECK_NE(x, kWriteInplace);
  }
  std::vector<NDArray> outputs;
  std::vector<OpReqType> req;
  extract_by_loc(_outputs, params.func_input_locs, &outputs);
  extract_by_loc(_req, params.func_input_locs, &req);
  if (state.n_iterations == 0) {
    for (int i = params.num_out_data; i < params.num_outputs; ++i) {
      int j = params.func_var_locs[i - params.num_out_data];
      mxnet::CopyFromTo(inputs[i], &outputs[j]);
    }
    state.Cleanup();
    return;
  }
  // collect var_locs and out_locs, positions other than var_locs are out_locs, i.e.
  // [0, var_locs[0])
  // (var_locs[1], var_locs[2]),
  // (var_locs[2], var_locs[3]),
  // ...
  // (var_locs[-2], var_locs[-1] = params.num_args - 2)
  std::vector<dim_t> var_locs(params.func_var_locs.begin(), params.func_var_locs.end());
  var_locs.push_back((dim_t) params.num_args - 2U);
  sort(var_locs.begin(), var_locs.end());
  // vectors for the backward loop
  std::vector<NDArray> ograds(params.num_outputs);
  std::vector<NDArray> igrads(outputs.size());
  std::vector<OpReqType> iter_req(req.size());
  for (int i = params.num_out_data; i < params.num_outputs; ++i)
    ograds[i] = inputs[i];
  const int n_iter = state.n_iterations;
  for (int step = n_iter - 1; step >= 0; --step) {
    // ograds[ : num_out_data] = inputs[ : num_out_data][step]
    // ograds[num_out_data: ] is maintained in the end of each loop
    std::transform(std::begin(inputs),
                   std::begin(inputs) + params.num_out_data,
                   std::begin(ograds),
                   [step] (const NDArray &a) { return a.At(step); } );
    // igrads[i] =
    //    outputs[i]            (step == 0)
    //    outputs[i]            (step != 0 && i not in loop_var_locs)
    //    ArrayLike(outputs[i]) (step != 0 && i in loop_var_locs)
    // iter_req =
    //    kWriteTo              (step != 0           && i in loop_var_locs)
    //    req[i]                (step == 0           && i in loop_var_locs)
    //    kAddTo                (step != n_iters - 1 && i not in loop_var_locs)
    //    req[i]                (step == n_iters - 1 && i not in loop_var_locs)
    {
      size_t i = 0;
      for (size_t loc : var_locs) {
        for ( ; i < loc; ++i) {
          // locs other that var_locs
          igrads[i] = outputs[i];
          iter_req[i] = (step + 1 == n_iter || req[i] == kNullOp)
                      ? req[i]
                      : kAddTo;
        }
        if (i < (size_t) params.num_args - 2U) {
          // a var
          igrads[i] = (step == 0)
                    ? outputs[i]
                    : NDArray(outputs[i].shape(), outputs[i].ctx(), true, outputs[i].dtype());
          iter_req[i] = (step == 0 || req[i] == kNullOp)
                      ? req[i]
                      : kWriteTo;
          ++i;
        } else {
          break;
        }
      }
    }
    state.Backward(step, ograds, iter_req, igrads);
    for (int i = params.num_out_data; i < params.num_outputs; ++i) {
      size_t j = params.func_var_locs[i - params.num_out_data];
      ograds[i] = igrads[j];
    }
  }
  state.Cleanup();
}

static bool WhileLoopShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape) {
  using nnvm::ShapeVector;
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  static const std::function<bool(const TShape &)> is_udf = is_shape_udf;
  // sanity checks
  CHECK_EQ(in_shape->size() + 2U, (size_t) params.num_args);
  CHECK_EQ(out_shape->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 2U);
  CHECK_EQ(attrs.subgraphs[0]->outputs.size(), 1U);
  // infer shape for cond and func
  auto infer_subg = [&params, in_shape, out_shape](std::shared_ptr<Symbol> subg,
                                                   ShapeVector *_subg_out,
                                                   const nnvm::Tuple<dim_t> &input_locs,
                                                   int num_out_data,
                                                   bool fill_out_shape) {
    // create subg_in
    ShapeVector subg_in;
    ShapeVector &subg_out = *_subg_out;
    extract_by_loc(*in_shape, input_locs, &subg_in);
    // create an indexed graph
    nnvm::Graph g;
    g.outputs = subg->outputs;
    const auto& idx = g.indexed_graph();
    // get input nodes
    const auto &input_nids = idx.input_nodes();
    // sanity checks
    CHECK_EQ(input_nids.size(), subg_in.size());
    CHECK_EQ(g.outputs.size(), subg_out.size());
    CHECK_EQ(idx.input_nodes().size(), subg_in.size());
    CHECK_EQ(idx.outputs().size(), subg_out.size());
    // create empty shapes for inference
    ShapeVector shapes(idx.num_node_entries());
    // copy subg_in into shapes
    for (size_t i = 0; i < subg_in.size(); ++i) {
      auto eid = idx.entry_id(input_nids[i], 0);
      shapes[eid] = subg_in[i];
    }
    // copy subg_out into shapes
    // note that ndim of out_data is not increased
    // because subg is only one step
    for (size_t i = 0; i < subg_out.size(); ++i) {
      auto eid = idx.entry_id(g.outputs[i]);
      shapes[eid] = subg_out[i];
    }
    // copy done, call InferShape
    g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
    g = exec::InferShape(std::move(g));
    // now `shapes' won't be used anymore, use new_shapes instead
    const auto& new_shapes = g.GetAttr<ShapeVector>("shape");
    // copy subg_in back to in_shape
    for (size_t i = 0; i < subg_in.size(); ++i) {
      auto eid = idx.entry_id(input_nids[i], 0);
      auto g_out_shape = new_shapes[eid];
      if (g_out_shape.ndim() == 0 || g_out_shape.Size() == 0) {
        // when the shape is not fully inferred
        continue;
      }
      SHAPE_ASSIGN_CHECK(*in_shape, input_locs[i], g_out_shape);
    }
    if (!fill_out_shape) {
      return true;
    }
    // copy subg_out back to out_shape
    // for results in [0, num_out_data), ndim should increase by 1
    for (int i = 0; i < num_out_data; ++i) {
      auto eid = idx.entry_id(g.outputs[i]);
      auto g_out_shape = new_shapes[eid];
      if (g_out_shape.ndim() == 0 || g_out_shape.Size() == 0) {
        // when the shape is not fully inferred
        continue;
      }
      auto out = TShape(g_out_shape.ndim() + 1);
      out[0] = params.max_iterations;
      for (size_t i = 1; i < out.ndim(); i++)
        out[i] = g_out_shape[i - 1];
      SHAPE_ASSIGN_CHECK(*out_shape, i, out);
    }
    // for results in [num_out_data, ...), ndim does not change
    for (size_t i = num_out_data; i < g.outputs.size(); ++i) {
      auto eid = idx.entry_id(g.outputs[i]);
      auto g_out_shape = new_shapes[eid];
      if (g_out_shape.ndim() == 0 || g_out_shape.Size() == 0) {
        // when the shape is not fully inferred
        continue;
      }
      SHAPE_ASSIGN_CHECK(*out_shape, i, g_out_shape);
    }
    return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
  };
  ShapeVector cond_out_shape{TShape(1U)};  // this means: [(1, )]
  ShapeVector func_out_shape(params.num_outputs);
  CHECK(params.sync_in_out(in_shape, out_shape, is_udf));
  bool succ_0 = infer_subg(attrs.subgraphs[0], &cond_out_shape, params.cond_input_locs, 0, false);
  CHECK(params.sync_in_out(in_shape, out_shape, is_udf));
  bool succ_1 = infer_subg(attrs.subgraphs[1], &func_out_shape, \
                           params.func_input_locs, params.num_out_data, true);
  CHECK(params.sync_in_out(in_shape, out_shape, is_udf));
  return succ_0 && succ_1;
}

static bool WhileLoopType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_type, std::vector<int> *out_type) {
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  static const std::function<bool(const int &)> is_udf = is_type_udf;
  CHECK_EQ(in_type->size() + 2U, (size_t) params.num_args);
  CHECK_EQ(out_type->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 2U);
  CHECK_EQ(attrs.subgraphs[0]->outputs.size(), 1U);
  std::vector<int> cond_in_type;
  std::vector<int> func_in_type;
  extract_by_loc(*in_type, params.cond_input_locs, &cond_in_type);
  extract_by_loc(*in_type, params.func_input_locs, &func_in_type);
  std::vector<int> cond_out_type = {0};
  CHECK(params.sync_in_out(in_type, out_type, is_udf));
  bool succ_0 = InferSubgraphDataType(*attrs.subgraphs[0], &cond_in_type, &cond_out_type);
  CHECK(params.sync_in_out(in_type, out_type, is_udf));
  CHECK(sync_in_in(params.cond_input_locs, in_type, &cond_in_type, is_udf));
  bool succ_1 = InferSubgraphDataType(*attrs.subgraphs[1], &func_in_type, out_type);
  CHECK(params.sync_in_out(in_type, out_type, is_udf));
  CHECK(sync_in_in(params.func_input_locs, in_type, &func_in_type, is_udf));
  return succ_0 && succ_1;
}

static bool WhileLoopStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  static const std::function<bool(const int &)> is_udf = is_stype_udf;
  CHECK_EQ(in_attrs->size() + 2U, (size_t) params.num_args);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 2U);
  CHECK_EQ(attrs.subgraphs[0]->outputs.size(), 1U);
  std::vector<int> cond_in_attrs;
  std::vector<int> func_in_attrs;
  extract_by_loc(*in_attrs, params.cond_input_locs, &cond_in_attrs);
  extract_by_loc(*in_attrs, params.func_input_locs, &func_in_attrs);
  std::vector<int> cond_out_attrs = {kDefaultStorage};
  DispatchMode cond_mode = DispatchMode::kUndefined;
  DispatchMode func_mode = DispatchMode::kUndefined;
  *dispatch_mode = DispatchMode::kFComputeEx;
  CHECK(params.sync_in_out(in_attrs, out_attrs, is_udf));
  bool succ_0 = InferSubgraphStorage(*attrs.subgraphs[0], dev_mask, \
                                     &cond_mode, &cond_in_attrs, &cond_out_attrs);
  CHECK(params.sync_in_out(in_attrs, out_attrs, is_udf));
  CHECK(sync_in_in(params.cond_input_locs, in_attrs, &cond_in_attrs, is_udf));
  bool succ_1 = InferSubgraphStorage(*attrs.subgraphs[1], dev_mask, \
                                     &func_mode, &func_in_attrs, out_attrs);
  CHECK(params.sync_in_out(in_attrs, out_attrs, is_udf));
  CHECK(sync_in_in(params.func_input_locs, in_attrs, &func_in_attrs, is_udf));
  return succ_0 && succ_1;
}

static bool BackwardWhileLoopStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  // `cond' is not backwarded, don't check
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size() + 2U, (size_t) params.num_args);
  CHECK_EQ(attrs.subgraphs.size(), 2U);
  CachedOp op(*attrs.subgraphs[1], {});
  return op.BackwardStorageType(attrs, dev_mask, dispatch_mode,
                                in_attrs, out_attrs);
}

static OpStatePtr CreateWhileLoopState(const NodeAttrs& attrs,
                                       Context ctx,
                                       const std::vector<TShape>& ishape,
                                       const std::vector<int>& itype) {
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  return OpStatePtr::Create<WhileLoopState>(params, *attrs.subgraphs[0], *attrs.subgraphs[1]);
}

static std::vector<nnvm::NodeEntry>
WhileLoopGradient(const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
  ElemwiseGradUseInOut fgrad{"_backward_while_loop"};
  std::vector<nnvm::NodeEntry> entries = fgrad(n, ograds);
  entries[0].node->attrs.subgraphs = n->attrs.subgraphs;
  return entries;
}

struct CondParam : public dmlc::Parameter<CondParam> {
  int num_args;
  int num_outputs;
  nnvm::Tuple<dim_t> cond_input_locs;
  nnvm::Tuple<dim_t> then_input_locs;
  nnvm::Tuple<dim_t> else_input_locs;
  DMLC_DECLARE_PARAMETER(CondParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(3)
    .describe("Number of input arguments, including cond, then and else as three symbol inputs.");
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("The number of outputs of the subgraph.");
    DMLC_DECLARE_FIELD(cond_input_locs)
    .describe("The locations of cond's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(then_input_locs)
    .describe("The locations of then's inputs in the given inputs.");
    DMLC_DECLARE_FIELD(else_input_locs)
    .describe("The locations of else's inputs in the given inputs.");
  }
};  // struct CondParam

DMLC_REGISTER_PARAMETER(CondParam);

class CondState {
 public:
  CondParam params;
  CachedOpPtr cond_op;
  LoopState then_branch;
  LoopState else_branch;
  int branch_selection;  // 1 if then branch; 0 if else branch; -1 if undefined

  CondState(const CondParam &params,
            const Symbol &cond,
            const Symbol &then_sym,
            const Symbol &else_sym):
            params(params),
            cond_op(LoopState::MakeSharedOp(cond)),
            then_branch(then_sym),
            else_branch(else_sym),
            branch_selection(-1) {
  }
};

static void CondComputeExCPU(const OpStatePtr& state_ptr,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  // The argument `inputs' are loop_vars and other inputs
  // loop_vars are stored in stored in `loop_vars_locs'
  // The argument `outputs' are output and new_loop_vars
  // [0: num_out_data) are outputs at each step.
  // [num_out_data: ) are new_loop_vars
  CondState &state = state_ptr.get_state<CondState>();
  const CondParam& params = state.params;
  // a helper function, converting std::vector<NDArray> to std::vector<NDArray*>
  const auto to_ptr_vec = [](std::vector<NDArray> &in, std::vector<NDArray*> *out) {
    out->clear();
    out->reserve(in.size());
    std::transform(std::begin(in),
                   std::end(in),
                   std::back_inserter(*out),
                   [](NDArray &a) {return &a;});
  };
  // sanity checks
  CHECK_EQ(inputs.size() + 3U, (size_t) params.num_args);
  CHECK_EQ(outputs.size(), (size_t) params.num_outputs);
  CHECK_EQ(outputs.size(), req.size());
  // construct inputs and outputs for cond
  std::vector<NDArray> cond_inputs;
  std::vector<NDArray> cond_outputs = {NDArray()};
  std::vector<NDArray*> cond_input_ptr;
  std::vector<NDArray*> cond_output_ptr;
  extract_by_loc(inputs, params.cond_input_locs, &cond_inputs);
  to_ptr_vec(cond_inputs, &cond_input_ptr);
  to_ptr_vec(cond_outputs, &cond_output_ptr);
  int &branch_selection = state.branch_selection;
  // run cond
  state.cond_op->Forward(nullptr, cond_input_ptr, cond_output_ptr);
  branch_selection = as_bool_scalar(*cond_output_ptr[0]);
  // select the right branch
  const nnvm::Tuple<dim_t> &func_input_locs = branch_selection
                                            ? params.then_input_locs
                                            : params.else_input_locs;
  LoopState &loop_state = branch_selection
                        ? state.then_branch
                        : state.else_branch;
  // extract inputs for the branch
  std::vector<NDArray> func_inputs;
  extract_by_loc(inputs, func_input_locs, &func_inputs);
  loop_state.Forward(0, func_inputs, req, outputs, ctx.need_grad);
}

static void CondGradComputeExCPU(const OpStatePtr& state_ptr,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& _req,
                                 const std::vector<NDArray>& outputs) {
  CondState &state = state_ptr.get_state<CondState>();
  const CondParam& params = state.params;
  // sanity checks
  CHECK_EQ(outputs.size() + 3U, (size_t) params.num_args);
  CHECK_EQ(outputs.size(), _req.size());
  // select the right branch
  int branch_selection = state.branch_selection;
  CHECK_NE(branch_selection, -1);
  const nnvm::Tuple<dim_t> &func_input_locs = branch_selection
                                            ? params.then_input_locs
                                            : params.else_input_locs;
  LoopState &loop_state = branch_selection
                        ? state.then_branch
                        : state.else_branch;
  // construct parameters
  std::vector<NDArray> ograds(inputs.begin(), inputs.begin() + params.num_outputs);
  std::vector<OpReqType> req;
  extract_by_loc(_req, func_input_locs, &req);
  std::vector<NDArray> igrads;
  extract_by_loc(outputs, func_input_locs, &igrads);
  loop_state.Backward(0, ograds, req, igrads);
  loop_state.Cleanup();
}

static bool CondShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_shape,
                      std::vector<TShape> *out_shape) {
  using nnvm::ShapeVector;
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  static const std::function<bool(const TShape &)> is_udf = is_shape_udf;
  // sanity checks
  CHECK_EQ(in_shape->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(out_shape->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  CHECK_EQ(attrs.subgraphs[0]->outputs.size(), 1U);
  CHECK_EQ(attrs.subgraphs[1]->outputs.size(), attrs.subgraphs[2]->outputs.size());
  // infer shape for cond, then and else
  auto infer_subg = [&params, in_shape, out_shape](std::shared_ptr<Symbol> subg,
                                                   ShapeVector *_subg_out,
                                                   const nnvm::Tuple<dim_t> &input_locs,
                                                   bool fill_out_shape) {
    // create subg_in
    ShapeVector subg_in;
    ShapeVector &subg_out = *_subg_out;
    extract_by_loc(*in_shape, input_locs, &subg_in);
    // create an indexed graph
    nnvm::Graph g;
    g.outputs = subg->outputs;
    const auto& idx = g.indexed_graph();
    // get input nodes
    const auto &input_nids = idx.input_nodes();
    // sanity checks
    CHECK_EQ(input_nids.size(), subg_in.size());
    CHECK_EQ(g.outputs.size(), subg_out.size());
    CHECK_EQ(idx.input_nodes().size(), subg_in.size());
    CHECK_EQ(idx.outputs().size(), subg_out.size());
    // create empty shapes for inference
    ShapeVector shapes(idx.num_node_entries());
    // copy subg_in into shapes
    for (size_t i = 0; i < subg_in.size(); ++i) {
      auto eid = idx.entry_id(input_nids[i], 0);
      shapes[eid] = subg_in[i];
    }
    // copy subg_out into shapes
    for (size_t i = 0; i < subg_out.size(); ++i) {
      auto eid = idx.entry_id(g.outputs[i]);
      shapes[eid] = subg_out[i];
    }
    // copy done, call InferShape
    g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
    g = exec::InferShape(std::move(g));
    // now `shapes' won't be used anymore, use new_shapes instead
    const auto& new_shapes = g.GetAttr<ShapeVector>("shape");
    // copy subg_in back to in_shape
    for (size_t i = 0; i < subg_in.size(); ++i) {
      auto eid = idx.entry_id(input_nids[i], 0);
      auto g_out_shape = new_shapes[eid];
      if (g_out_shape.ndim() == 0 || g_out_shape.Size() == 0) {
        // when the shape is not fully inferred
        continue;
      }
      SHAPE_ASSIGN_CHECK(*in_shape, input_locs[i], g_out_shape);
    }
    if (!fill_out_shape) {
      return true;
    }
    // copy subg_out back to out_shape
    for (size_t i = 0; i < g.outputs.size(); ++i) {
      auto eid = idx.entry_id(g.outputs[i]);
      auto g_out_shape = new_shapes[eid];
      if (g_out_shape.ndim() == 0 || g_out_shape.Size() == 0) {
        // when the shape is not fully inferred
        continue;
      }
      SHAPE_ASSIGN_CHECK(*out_shape, i, g_out_shape);
    }
    return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
  };
  ShapeVector cond_out_shape{TShape(1U)};  // this means: [(1, )]
  ShapeVector then_out_shape(params.num_outputs);
  ShapeVector else_out_shape(params.num_outputs);
  bool succ_0 = infer_subg(attrs.subgraphs[0], &cond_out_shape, \
                           params.cond_input_locs, false);
  bool succ_1 = infer_subg(attrs.subgraphs[1], &then_out_shape, \
                           params.then_input_locs, true);
  bool succ_2 = infer_subg(attrs.subgraphs[2], &else_out_shape, \
                           params.else_input_locs, true);
  sync_out_out(&then_out_shape, &else_out_shape, is_udf);
  return succ_0 && succ_1 && succ_2;
}

static bool CondType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_type,
                     std::vector<int> *out_type) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  static const std::function<bool(const int &)> is_udf = is_type_udf;
  CHECK_EQ(in_type->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(out_type->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  CHECK_EQ(attrs.subgraphs[0]->outputs.size(), 1U);
  CHECK_EQ(attrs.subgraphs[1]->outputs.size(), attrs.subgraphs[2]->outputs.size());
  std::vector<int> cond_in_type;
  std::vector<int> then_in_type;
  std::vector<int> else_in_type;
  extract_by_loc(*in_type, params.cond_input_locs, &cond_in_type);
  extract_by_loc(*in_type, params.then_input_locs, &then_in_type);
  extract_by_loc(*in_type, params.else_input_locs, &else_in_type);
  std::vector<int> cond_out_type = {0};
  bool succ_0 = InferSubgraphDataType(*attrs.subgraphs[0], &cond_in_type, &cond_out_type);
  CHECK(sync_in_in(params.cond_input_locs, in_type, &cond_in_type, is_udf));
  bool succ_1 = InferSubgraphDataType(*attrs.subgraphs[1], &then_in_type, out_type);
  CHECK(sync_in_in(params.then_input_locs, in_type, &then_in_type, is_udf));
  bool succ_2 = InferSubgraphDataType(*attrs.subgraphs[2], &else_in_type, out_type);
  CHECK(sync_in_in(params.else_input_locs, in_type, &else_in_type, is_udf));
  return succ_0 && succ_1 && succ_2;
}

static bool CondStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  static const std::function<bool(const int &)> is_udf = is_stype_udf;
  CHECK_EQ(in_attrs->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  CHECK_EQ(attrs.subgraphs[0]->outputs.size(), 1U);
  CHECK_EQ(attrs.subgraphs[1]->outputs.size(), attrs.subgraphs[2]->outputs.size());
  std::vector<int> cond_in_attrs;
  std::vector<int> then_in_attrs;
  std::vector<int> else_in_attrs;
  extract_by_loc(*in_attrs, params.cond_input_locs, &cond_in_attrs);
  extract_by_loc(*in_attrs, params.then_input_locs, &then_in_attrs);
  extract_by_loc(*in_attrs, params.else_input_locs, &else_in_attrs);
  std::vector<int> cond_out_attrs = {kDefaultStorage};
  DispatchMode cond_mode = DispatchMode::kUndefined;
  DispatchMode then_mode = DispatchMode::kUndefined;
  DispatchMode else_mode = DispatchMode::kUndefined;
  *dispatch_mode = DispatchMode::kFComputeEx;
  bool succ_0 = InferSubgraphStorage(*attrs.subgraphs[0], dev_mask, \
                                     &cond_mode, &cond_in_attrs, &cond_out_attrs);
  CHECK(sync_in_in(params.cond_input_locs, in_attrs, &cond_in_attrs, is_udf));
  bool succ_1 = InferSubgraphStorage(*attrs.subgraphs[1], dev_mask, \
                                     &then_mode, &then_in_attrs, out_attrs);
  CHECK(sync_in_in(params.then_input_locs, in_attrs, &then_in_attrs, is_udf));
  bool succ_2 = InferSubgraphStorage(*attrs.subgraphs[2], dev_mask, \
                                     &else_mode, &else_in_attrs, out_attrs);
  CHECK(sync_in_in(params.else_input_locs, in_attrs, &else_in_attrs, is_udf));
  return succ_0 && succ_1 && succ_2;
}

static bool BackwardCondStorageType(const nnvm::NodeAttrs& attrs,
                                    const int dev_mask,
                                    DispatchMode* dispatch_mode,
                                    std::vector<int> *in_attrs,
                                    std::vector<int> *out_attrs) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  static const std::function<bool(const int &)> is_udf = is_stype_udf;
  auto sub_pass = [&](const std::shared_ptr<Symbol> &subg, const nnvm::Tuple<dim_t> &input_locs) {
    // A. first construct subg_in_attrs
    // need subg_in_attrs as subg_bwd_out (copy), subg_fwd_in (extract), subg_fwd_out (copy)
    std::vector<int> subg_in_attrs;
    size_t num_elts = params.num_outputs * 2 + input_locs.ndim();
    subg_in_attrs.reserve(num_elts);
    // part 1. subg_bwd_out (copy)
    subg_in_attrs.insert(subg_in_attrs.end(),
                         in_attrs->begin(),
                         in_attrs->begin() + params.num_outputs);
    // part 2. subg_fwd_in (extract)
    std::vector<int> fwd_in(in_attrs->begin() + params.num_outputs,
                            in_attrs->begin() + params.num_outputs + params.num_args - 3);
    std::vector<int> subg_fwd_in;
    extract_by_loc(fwd_in, input_locs, &subg_fwd_in);
    subg_in_attrs.insert(subg_in_attrs.end(),
                         subg_fwd_in.begin(),
                         subg_fwd_in.end());
    // part 3. subg_fwd_out (copy)
    subg_in_attrs.insert(subg_in_attrs.end(),
                         in_attrs->begin() + params.num_outputs + params.num_args - 3,
                         in_attrs->end());
    // check correctness of the number of elements
    CHECK_EQ(subg_in_attrs.size(), num_elts);
    // B. then we construct subg_out_attrs by extracting from out_attrs
    std::vector<int> subg_out_attrs;
    extract_by_loc(*out_attrs, input_locs, &subg_out_attrs);
    // then we construct the subgraph and do inference
    CachedOp op(*subg, {});
    bool ret = op.BackwardStorageType(attrs, dev_mask, dispatch_mode, \
                                      &subg_in_attrs, &subg_out_attrs);
    CHECK(sync_in_in(input_locs, out_attrs, &subg_out_attrs, is_udf));
    return ret;
  };
  for (const dim_t &cond_in : params.cond_input_locs) {
    (*out_attrs)[cond_in] = kDefaultStorage;
  }
  bool succ_0 = sub_pass(attrs.subgraphs[1], params.then_input_locs);
  bool succ_1 = sub_pass(attrs.subgraphs[2], params.else_input_locs);
  return succ_0 && succ_1;
}

static OpStatePtr CreateCondState(const NodeAttrs& attrs,
                                  Context ctx,
                                  const std::vector<TShape>& ishape,
                                  const std::vector<int>& itype) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  return OpStatePtr::Create<CondState>(
    params,
    *attrs.subgraphs[0],
    *attrs.subgraphs[1],
    *attrs.subgraphs[2]);
}

static std::vector<nnvm::NodeEntry>
CondGradient(const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
  ElemwiseGradUseInOut fgrad{"_backward_cond"};
  std::vector<nnvm::NodeEntry> entries = fgrad(n, ograds);
  entries[0].node->attrs.subgraphs = n->attrs.subgraphs;
  return entries;
}

NNVM_REGISTER_OP(_foreach)
.MXNET_DESCRIBE("Run a for loop over an NDArray with user-defined computation")
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
// Foreach operator works like an executor. Its code will always run on CPU.
// So the same code can be registered for both CPU and GPU.
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", ForeachComputeExCPU)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
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
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
.set_attr<FInferStorageType>("FInferStorageType", BackwardForeachStorageType)
.set_attr_parser(ParamParser<ForeachParam>)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ForeachGradComputeExCPU)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", ForeachGradComputeExCPU);

NNVM_REGISTER_OP(_while_loop)
.MXNET_DESCRIBE("Run a while loop over with user-defined condition and computation")
.set_attr_parser(ParamParser<WhileLoopParam>)
.set_attr<FInferStorageType>("FInferStorageType", WhileLoopStorageType)
.set_num_inputs([](const NodeAttrs& attrs) {
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  return params.num_outputs;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  std::vector<std::string> names;
  names.reserve(params.num_args);
  names.push_back("cond");
  names.push_back("func");
  for (int i = 2; i < params.num_args; i++)
    names.push_back("data" + std::to_string(i - 2));
  return names;
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return std::vector<uint32_t>{0, 1};
})
.set_attr<nnvm::FGradient>("FGradient", WhileLoopGradient)
.set_attr<FCreateOpState>("FCreateOpState", CreateWhileLoopState)
.set_attr<nnvm::FInferShape>("FInferShape", WhileLoopShape)
.set_attr<nnvm::FInferType>("FInferType", WhileLoopType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", WhileLoopComputeExCPU)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", WhileLoopComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("cond", "Symbol", "Input graph for the loop condition.")
.add_argument("func", "Symbol", "Input graph for the loop body.")
.add_argument("data", "NDArray-or-Symbol[]",
              "The input arrays that include data arrays and states.")
.add_arguments(WhileLoopParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_while_loop)
.set_num_inputs([](const NodeAttrs& attrs){
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  return params.num_outputs * 2 + params.num_args - 2;
})
.set_num_outputs([](const NodeAttrs& attrs){
  const WhileLoopParam& params = nnvm::get<WhileLoopParam>(attrs.parsed);
  return params.num_args - 2;
})
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
.set_attr<FInferStorageType>("FInferStorageType", BackwardWhileLoopStorageType)
.set_attr_parser(ParamParser<WhileLoopParam>)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", WhileLoopGradComputeExCPU)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", WhileLoopGradComputeExCPU);

NNVM_REGISTER_OP(_cond)
.MXNET_DESCRIBE("Run a if-then-else using user-defined condition and computation")
.set_attr_parser(ParamParser<CondParam>)
.set_attr<FInferStorageType>("FInferStorageType", CondStorageType)
.set_num_inputs([](const NodeAttrs& attrs) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  return params.num_outputs;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  std::vector<std::string> names;
  names.reserve(params.num_args);
  names.push_back("cond");
  names.push_back("then_branch");
  names.push_back("else_branch");
  for (int i = 3; i < params.num_args; ++i)
    names.push_back("data" + std::to_string(i - 3));
  return names;
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return std::vector<uint32_t>{0, 1, 2};
})
.set_attr<nnvm::FGradient>("FGradient", CondGradient)
.set_attr<FCreateOpState>("FCreateOpState", CreateCondState)
.set_attr<nnvm::FInferShape>("FInferShape", CondShape)
.set_attr<nnvm::FInferType>("FInferType", CondType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CondComputeExCPU)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CondComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("cond", "Symbol", "Input graph for the condition.")
.add_argument("then_branch", "Symbol", "Input graph for the then branch.")
.add_argument("else_branch", "Symbol", "Input graph for the else branch.")
.add_argument("data", "NDArray-or-Symbol[]",
              "The input arrays that include data arrays and states.")
.add_arguments(CondParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_cond)
.set_num_inputs([](const NodeAttrs& attrs){
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  return params.num_outputs * 2 + params.num_args - 3;
})
.set_num_outputs([](const NodeAttrs& attrs){
  const CondParam& params = nnvm::get<CondParam>(attrs.parsed);
  return params.num_args - 3;
})
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
.set_attr<FInferStorageType>("FInferStorageType", BackwardCondStorageType)
.set_attr_parser(ParamParser<CondParam>)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CondGradComputeExCPU)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CondGradComputeExCPU);
}  // namespace op
}  // namespace mxnet
