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

struct SubgraphOpState {
  // TODO: initialize uuid
  SubgraphOpState(const Symbol& sym) :
      subgraph_sym_(&sym),
      subgraph_uuid_("dfasdfadsmxdfw324"),
      immutable_data_names_(sym.ListInputNames(Symbol::kReadOnlyArgs)),
      mutable_data_names_(sym.ListInputNames(Symbol::kAuxiliaryStates)),
      //input_data_names_(sym.ListInputNames(Symbol::kAll)),
      output_data_names_(sym.ListOutputNames()),
      subgraph_executor_(nullptr) {
    const std::vector<std::string> input_data_names = sym.ListInputNames(Symbol::kAll);
    //const std::vector<std::string> immutable_data_names = sym.ListInputNames(Symbol::kReadOnlyArgs);
    //const std::vector<std::string> mutable_data_names = sym.ListInputNames(Symbol::kAuxiliaryStates);
    immutable_data_indices_.resize(immutable_data_names_.size());
    mutable_data_indices_.resize(mutable_data_names_.size());
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
  }

	// arguments should have the same order as NDArrays in FCompute
  const Symbol* subgraph_sym_;
  // this variable records the NDArrays' var versions of the last run.
  std::vector<int64_t> ndarray_var_versions_;
  std::vector<uint32_t> immutable_data_indices_;
  std::vector<uint32_t> mutable_data_indices_;
  std::vector<std::string> immutable_data_names_;
  std::vector<std::string> mutable_data_names_;
  //std::vector<std::string> input_data_names_;
  std::vector<std::string> output_data_names_;
  std::string subgraph_uuid_;
  std::shared_ptr<Executor> subgraph_executor_;
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
  // We can create an executor to run this subgraph op
  if (state.subgraph_executor_.get() == nullptr) {
    std::vector<NDArray> arg_arrays;
    std::vector<NDArray> aux_arrays;
    for (size_t i = 0; i < state.immutable_data_indices_.size(); ++i) {
      arg_arrays.push_back(inputs[state.immutable_data_indices_[i]]);
    }
    for (size_t i = 0; i < state.mutable_data_indices_.size(); ++i) {
      aux_arrays.push_back(inputs[state.mutable_data_indices_[i]]);
    }
    std::vector<NDArray> grad_store(arg_arrays.size());
    std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);
    state.subgraph_executor_.reset(Executor::Bind(*state.subgraph_sym_,
          ctx.run_ctx.ctx, std::map<std::string, Context>(), arg_arrays, grad_store,
          grad_req, aux_arrays));
  }
  // TODO: replace the hard-coded integer with inputs[i].var().version
  // If var version is old, need to update ndarray in the executor.
  const int64_t max_var_version = 12034324324;
  for (size_t i = 0; i < state.immutable_data_names_.size(); ++i) {
    if (state.ndarray_var_versions_[state.immutable_data_indices_[i]] < max_var_version) {
      auto it = state.subgraph_executor_->in_arg_map().find(state.immutable_data_names_[i]);
      CHECK(it != state.subgraph_executor_->in_arg_map().end());
      // Commented out because we don't have interface to do it yet
      it->second = inputs[state.immutable_data_indices_[i]];
      ++state.ndarray_var_versions_[state.immutable_data_indices_[i]];
    }
  }
  for (size_t i = 0; i < state.mutable_data_names_.size(); ++i) {
    if (state.ndarray_var_versions_[state.mutable_data_indices_[i]] < max_var_version) {
      auto it = state.subgraph_executor_->aux_state_map().find(state.mutable_data_names_[i]);
      CHECK(it != state.subgraph_executor_->aux_state_map().end());
      // Commented out because we don't have interface to do it yet
      it->second = inputs[state.mutable_data_indices_[i]];
      ++state.ndarray_var_versions_[state.mutable_data_indices_[i]];
    }
  }
  state.subgraph_executor_->Forward(false);
  for (size_t i = 0; i < outputs.size(); ++i) {
    //NDArray tmp = outputs[i];
    //state.subgraph_executor_->output_arrays()[i].WaitToRead();
    CopyFromTo(state.subgraph_executor_->output_arrays()[i], &outputs[i]);
    //tmp = state.subgraph_executor_->output_arrays()[i];
  }

#if 0
  for (const auto& name : state.immutable_data_names_) {
    LOG(INFO) << "SubgraphOpForward: input_data_name = " << name;
  }
  for (const auto& name : state.output_data_names_) {
    LOG(INFO) << "SubgraphOpForward: output_data_name = " << name;
  }
  for (const auto v : state.ndarray_var_versions_) {
    LOG(INFO) << "SubgraphOpForward: var_version = " << v;
  }
#endif
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
