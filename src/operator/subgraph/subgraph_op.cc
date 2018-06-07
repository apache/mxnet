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
#include "../../imperative/cached_op.h"
#include "./subgraph_op.h"

namespace mxnet {
namespace op {

class DefaultSubgraphOperator {
 public:
  DefaultSubgraphOperator(const Symbol& sym) :
      //subgraph_uuid_("dfasdfadsmxdfw324"),
      //immutable_data_names_(sym.ListInputNames(Symbol::kReadOnlyArgs)),
      //mutable_data_names_(sym.ListInputNames(Symbol::kAuxiliaryStates)),
      subgraph_sym_(sym),
      input_names_(sym.ListInputNames(Symbol::kAll)),
      output_names_(sym.ListOutputNames()) {
    //subgraph_exec_.reset(new CachedOp(sym, {{"static_alloc", "true"}}));
    subgraph_exec_.reset(new CachedOp(sym, {}));
    //const std::vector<std::string> input_data_names = sym.ListInputNames(Symbol::kAll);
    //const std::vector<std::string> immutable_data_names = sym.ListInputNames(Symbol::kReadOnlyArgs);
    //const std::vector<std::string> mutable_data_names = sym.ListInputNames(Symbol::kAuxiliaryStates);
    //immutable_data_indices_.resize(immutable_data_names_.size());
    //mutable_data_indices_.resize(mutable_data_names_.size());
#if 0
    for (uint32_t i = 0, j1 = 0, j2 = 0; i < input_data_names.size(); ++i) {
      if (input_data_names[i] == immutable_data_names_[j1]) {
        immutable_data_indices_[j1++] = i;
      } else if (input_data_names[i] == mutable_data_names_[j2]) {
        mutable_data_indices_[j2++] = i;
      } else {
        LOG(FATAL) << "Should not happen";
      }
    }
    // initialize var versions to -1
    ndarray_var_versions_.resize(input_data_names.size(), -1);
#endif
  }

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);
  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented";
  }

 private:
  nnvm::Symbol subgraph_sym_;
  //std::string subgraph_uuid_;
  // this variable records the NDArrays' var versions of the last run.
  //std::vector<uint32_t> immutable_data_indices_;
  //std::vector<uint32_t> mutable_data_indices_;
  //std::vector<std::string> immutable_data_names_;
  //std::vector<std::string> mutable_data_names_;
  //std::vector<std::string> input_data_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  //std::vector<int64_t> ndarray_var_versions_;
  //std::shared_ptr<Executor> subgraph_executor_;
  CachedOpPtr subgraph_exec_;
};

void DefaultSubgraphOperator::Forward(const OpContext& ctx,
                                      const std::vector<NDArray>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<NDArray>& outputs) {
  std::vector<NDArray> tmp_inputs = inputs;
  std::vector<NDArray*> input_ptrs;
  input_ptrs.reserve(inputs.size());
  for (auto& nd : tmp_inputs) {
    input_ptrs.push_back(&nd);
  }
  std::vector<NDArray> tmp_outputs = outputs;
  std::vector<NDArray*> output_ptrs;
  for (auto& nd : tmp_outputs) {
    output_ptrs.push_back(&nd);
  }
  subgraph_exec_->Forward(subgraph_exec_, input_ptrs, output_ptrs);
}

OpStatePtr CreateSubgraphOpState(const NodeAttrs& attrs,
                                 Context ctx,
                                 const std::vector<TShape>& in_shapes,
                                 const std::vector<int>& in_types) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  return OpStatePtr::Create<DefaultSubgraphOperator>(subgraph_sym);
}

bool SubgraphOpShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_shapes,
                     std::vector<TShape> *out_shapes) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_shapes->size());
  CHECK_EQ(idx_g.outputs().size(), out_shapes->size());
  // TODO: make sure shape inputs matches the order from in_shapes

  // Put the input and output shapes to the shape vector.
  nnvm::ShapeVector shapes(idx_g.num_node_entries());
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_shapes->size());
  for (size_t i = 0; i < in_shapes->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    shapes[eid] = in_shapes->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_shapes->size());
  for (size_t i = 0; i < out_shapes->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    shapes[eid] = out_shapes->at(i);
  }

  // Infer shape of the graph.
  g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  g = exec::InferShape(std::move(g));

  // Copy the inferred shape back to the input shapes and the output shapes.
  shapes = g.GetAttr<nnvm::ShapeVector>("shape");
  // assign to in_shapes
  for (size_t i = 0; i < in_shapes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    SHAPE_ASSIGN_CHECK(*in_shapes, i, shapes[eid]);
  }
  // assign to out_shapes
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    SHAPE_ASSIGN_CHECK(*out_shapes, i, shapes[eid]);
  }
  // Check if we have inferred the shapes correctly.
  return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
}

bool SubgraphOpType(const nnvm::NodeAttrs& attrs,
                    std::vector<int> *in_types,
                    std::vector<int> *out_types) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_types->size());
  CHECK_EQ(idx_g.outputs().size(), out_types->size());
  // TODO: make sure type inputs matches the order from in_types

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

  types = g.GetAttr<nnvm::DTypeVector>("dtype");
  // assign to in_types
  for (size_t i = 0; i < in_types->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_types, i, types[eid]);
  }
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    TYPE_ASSIGN_CHECK(*out_types, i, types[eid]);
  }
  // Check if we have inferred the dtypes correctly.
  return g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0;
}

bool SubgraphOpStorageType(const nnvm::NodeAttrs& attrs,
                           const int dev_mask,
                           DispatchMode* dispatch_mode,
                           std::vector<int>* in_stypes,
                           std::vector<int>* out_stypes) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_stypes->size());
  CHECK_EQ(idx_g.outputs().size(), out_stypes->size());
  exec::DevMaskVector dev_masks(idx_g.num_node_entries(), dev_mask);
  // TODO: make sure type inputs matches the order from in_types

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

  stypes = g.GetAttr<StorageTypeVector>("storage_type");
  // assign to in_types
  for (size_t i = 0; i < in_stypes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, i, stypes[eid]);
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    STORAGE_TYPE_ASSIGN_CHECK(*out_stypes, i, stypes[eid]);
  }
  // Check if we have inferred the storages correctly.
  return g.GetAttr<size_t>("storage_type_num_unknown_nodes") == 0;
}

void SubgraphOpForward(const OpStatePtr& state_ptr,
                       const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) {
  DefaultSubgraphOperator& op = state_ptr.get_state<DefaultSubgraphOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(_subgraph_op)
.describe(R"code(_subgraph_op)code" ADD_FILELINE)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const Symbol& sym = nnvm::get<Symbol>(attrs.parsed);
    return sym.ListInputNames(Symbol::kAll).size();
  })
.set_num_outputs(
  [](const NodeAttrs& attrs) {
    const Symbol& sym = nnvm::get<Symbol>(attrs.parsed);
    return sym.ListOutputNames().size();
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const Symbol& sym = nnvm::get<Symbol>(attrs.parsed);
    return sym.ListInputNames(Symbol::kAll);
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    const Symbol& sym = nnvm::get<Symbol>(attrs.parsed);
    return sym.ListOutputNames();
  })
.set_attr<FCreateOpState>("FCreateOpState", CreateSubgraphOpState)
.set_attr<nnvm::FInferShape>("FInferShape", SubgraphOpShape)
.set_attr<nnvm::FInferType>("FInferType", SubgraphOpType)
.set_attr<FInferStorageType>("FInferStorageType", SubgraphOpStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SubgraphOpForward)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("data", "NDArray-or-Symbol[]", "input data list");

}  // namespace op
}  // namespace mxnet
