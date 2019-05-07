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

#include "./subgraph_op_common.h"
#include "./operator_common.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

bool InferSubgraphDataType(const nnvm::Symbol &subgraph,
                           std::vector<int> *in_types,
                           std::vector<int> *out_types) {
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_types->size());
  CHECK_EQ(idx_g.outputs().size(), out_types->size());

  // Put the input and output data types to the dtype vector.
  nnvm::DTypeVector types(idx_g.num_node_entries(), -1);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_types->size());
  for (size_t i = 0; i < in_types->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    types[eid] = in_types->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_types->size());
  for (size_t i = 0; i < out_types->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    types[eid] = out_types->at(i);
  }

  // Infer data type of the graph.
  g.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
  g = exec::InferType(std::move(g));

  const auto& types1 = g.GetAttr<nnvm::DTypeVector>("dtype");
  // assign to in_types
  for (size_t i = 0; i < in_types->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_types, i, types1[eid]);
  }
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    TYPE_ASSIGN_CHECK(*out_types, i, types1[eid]);
  }
  // Check if we have inferred the dtypes correctly.
  return g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0;
}

bool InferSubgraphStorage(const nnvm::Symbol &subgraph,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int> *in_stypes,
                          std::vector<int> *out_stypes) {
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_stypes->size());
  CHECK_EQ(idx_g.outputs().size(), out_stypes->size());
  exec::DevMaskVector dev_masks(idx_g.num_node_entries(), dev_mask);

  // Put the input and output storages to the storage vector.
  nnvm::StorageVector stypes(idx_g.num_node_entries(), exec::kBadStorageID);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_stypes->size());
  for (size_t i = 0; i < in_stypes->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    stypes[eid] = in_stypes->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_stypes->size());
  for (size_t i = 0; i < out_stypes->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    stypes[eid] = out_stypes->at(i);
  }

  // Infer storage type of the graph.
  bool dev_match = g.attrs.count("dev_mask") &&
                   g.GetAttr<exec::DevMaskVector>("dev_mask") == dev_masks;
  if (!dev_match) {
    g.attrs["dev_mask"] = std::make_shared<dmlc::any>(std::move(dev_masks));
  }
  g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(stypes));
  g = exec::InferStorageType(std::move(g));

  const auto& stypes1 = g.GetAttr<StorageTypeVector>("storage_type");
  // assign to in_types
  for (size_t i = 0; i < in_stypes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, i, stypes1[eid]);
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    STORAGE_TYPE_ASSIGN_CHECK(*out_stypes, i, stypes1[eid]);
  }
  // Check if we have inferred the storages correctly.
  return g.GetAttr<size_t>("storage_type_num_unknown_nodes") == 0;
}

bool InferSubgraphShape(const nnvm::Symbol &subgraph,
                        mxnet::ShapeVector *in_shape,
                        mxnet::ShapeVector *out_shape) {
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx = g.indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_shape->size());
  CHECK_EQ(idx.outputs().size(), out_shape->size());

  // Put the input and output shapes to the shape vector.
  mxnet::ShapeVector shapes(idx.num_node_entries());
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_shape->size());
  for (size_t i = 0; i < in_shape->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    shapes[eid] = in_shape->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_shape->size());
  for (size_t i = 0; i < out_shape->size(); i++) {
    auto eid = idx.entry_id(g.outputs[i]);
    shapes[eid] = out_shape->at(i);
  }

  // Infer shape of the graph.
  g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  g = exec::InferShape(std::move(g));

  const auto& shapes1 = g.GetAttr<mxnet::ShapeVector>("shape");
  // Inferring the shape in the subgraph may infer the shape of the inputs.
  // We need to copy the inferred input shapes back.
  CHECK_EQ(input_nids.size(), in_shape->size());
  for (size_t i = 0; i < in_shape->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    SHAPE_ASSIGN_CHECK(*in_shape, i, shapes1[eid]);
  }

  for (size_t i = 0; i < g.outputs.size(); i++) {
    uint32_t eid = idx.entry_id(g.outputs[i]);
    SHAPE_ASSIGN_CHECK(*out_shape, i, shapes1[eid]);
  }
  return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
}

template <typename T>
T _asscalar(const NDArray &a) {
  CHECK_EQ(a.shape().Size(), 1U);
  T data;
  a.SyncCopyToCPU(&data, 1U);
  return data;
}

bool as_bool_scalar(const NDArray &a) {
  MSHADOW_TYPE_SWITCH(a.dtype(), DType, {
    return static_cast<bool>(_asscalar<DType>(a));
  });
  LOG(FATAL) << "Unknown dtype";
  return false;
}

bool is_shape_udf(const mxnet::TShape &x) {
  return !shape_is_known(x);
}

bool is_stype_udf(const int &x) {
  return x == exec::kBadStorageID;
}

bool is_type_udf(const int &x) {
  return x == -1;
}

LoopState::LoopState(const Symbol &g) {
  this->subgraph_sym = g;
  this->subgraph.outputs = g.outputs;
  this->iter_op = LoopState::MakeSharedOp(g);
}

void LoopState::Forward(int iter_no,
                        const std::vector<NDArray> &cinputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray> &coutputs,
                        bool is_recording) {
  using namespace nnvm;
  using namespace imperative;

  bool orig_is_record;
  if (is_recording)
    orig_is_record = Imperative::Get()->set_is_recording(true);
  else
    orig_is_record = Imperative::Get()->is_recording();

  std::vector<NDArray> in_bufs = cinputs;
  std::vector<NDArray> out_bufs = coutputs;
  std::vector<NDArray *> inputs(cinputs.size());
  std::vector<NDArray *> outputs(coutputs.size());
  for (size_t i = 0; i < inputs.size(); i++)
    inputs[i] = &in_bufs[i];
  for (size_t i = 0; i < outputs.size(); i++)
    outputs[i] = &out_bufs[i];

  OpStatePtr state = iter_op->Forward(nullptr, inputs, outputs);
  // If an input and an output share the array, the output array will be changed
  // by CachedOp. We need to copy data to the real output.
  for (size_t i = 0; i < out_bufs.size(); i++)
    if (!out_bufs[i].IsSame(coutputs[i])) {
      // The line below checks whether dynamic shape exists.
      // If so, re-initialize the shape.
      if (!shape_is_known(coutputs[i].shape())) {
        const_cast<NDArray &>(coutputs[i]).Init(out_bufs[i].shape());
      }
      CopyFromTo(out_bufs[i], coutputs[i]);
    }
  if (is_recording) {
    all_inputs.push_back(cinputs);
    all_outputs.push_back(coutputs);
    all_states.push_back(state);
  }

  Imperative::Get()->set_is_recording(orig_is_record);
}

void LoopState::Backward(int iter_no,
                         const std::vector<NDArray> &ograds,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &igrads) {
  using namespace nnvm;
  using namespace imperative;

  CHECK_GT(all_states.size(), iter_no)
      << "We didn't record the computation for iteration " << iter_no;
  auto op = iter_op;
  std::vector<NDArray *> inputs;
  std::vector<NDArray *> outputs;
  inputs.reserve(op->num_backward_inputs());
  outputs.reserve(op->num_inputs());
  std::vector<NDArray> ograd_bufs = ograds;
  std::vector<NDArray> igrad_bufs = igrads;
  for (size_t i = 0; i < ograds.size(); i++)
    inputs.push_back(&ograd_bufs[i]);

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
    outputs.push_back(&igrad_bufs[i]);
  CHECK_EQ(outputs.size(), op->num_inputs());
  auto state = all_states[iter_no];
  op->Backward(false, state, inputs, req, outputs);
  // If an input and an output share the array, the output array will be changed
  // by CachedOp. We need to copy data to the real output.
  for (size_t i = 0; i < igrads.size(); i++)
    if (!igrads[i].IsSame(igrad_bufs[i]))
      CopyFromTo(igrad_bufs[i], igrads[i]);
}

}  // namespace op
}  // namespace mxnet
