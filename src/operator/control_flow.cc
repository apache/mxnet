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

}  // namespace op
}  // namespace mxnet
