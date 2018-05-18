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

class SubgraphOpState {
 public:
  // TODO: initialize uuid
  SubgraphOpState(const Symbol& sym) :
    subgraph_sym_(&sym),
    subgraph_uuid_("dfasdfadsmxdfw324"),
    input_data_names_(sym.ListInputNames(Symbol::kAll)),
    output_data_names_(sym.ListOutputNames()),
    emc_(nullptr) {
    // initialize var versions to -1
    var_versions_.resize(input_data_names_.size(), -1);
  }
	// should have the same order as NDArrays in FCompute
  const Symbol* subgraph_sym_;
  std::vector<int64_t> var_versions_;
  std::vector<std::string> input_data_names_;
  std::vector<std::string> output_data_names_;
  std::string subgraph_uuid_;
  //ModelContext* emc_;
  void* emc_;
};  // SubgraphOpState

OpStatePtr CreateSubgraphOpState(const NodeAttrs& attrs,
                                 Context ctx,
                                 const std::vector<TShape>& in_shapes,
                                 const std::vector<int>& in_types) {
  const Symbol& subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  return OpStatePtr::Create<SubgraphOpState>(subgraph_sym);
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
  nnvm::ShapeVector shape_inputs = *in_shapes;
  imperative::CheckAndInferShape(&g, std::move(shape_inputs), true);
  const nnvm::ShapeVector& shapes = g.GetAttr<nnvm::ShapeVector>("shape");
  const std::vector<uint32_t>& input_nids = idx_g.input_nodes();
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
  return true;
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
  nnvm::DTypeVector type_inputs = *in_types;
  imperative::CheckAndInferType(&g, std::move(type_inputs), true);
  const nnvm::DTypeVector& types = g.GetAttr<nnvm::DTypeVector>("dtype");
  const std::vector<uint32_t>& input_nids = idx_g.input_nodes();
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
  return true;
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
  exec::DevMaskVector dev_masks(idx_g.num_nodes(), dev_mask);
  // TODO: make sure type inputs matches the order from in_types
  StorageTypeVector stype_inputs = *in_stypes;
  imperative::CheckAndInferStorageType(&g, std::move(dev_masks),
                                                        std::move(stype_inputs), true);
  const StorageTypeVector& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  const std::vector<uint32_t>& input_nids = idx_g.input_nodes();
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
  return true;
}

void SubgraphOpForward(const OpStatePtr& state_ptr,
                       const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) {
  SubgraphOpState& state = state_ptr.get_state<SubgraphOpState>();
  for (const auto& name : state.input_data_names_) {
    LOG(INFO) << "SubgraphOpForward: input_data_name = " << name;
  }
  for (const auto& name : state.output_data_names_) {
    LOG(INFO) << "SubgraphOpForward: output_data_name = " << name;
  }
  for (const auto v : state.var_versions_) {
    LOG(INFO) << "SubgraphOpForward: var_version = " << v;
  }
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
