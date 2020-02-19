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

/*!
 *  Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief graph executor
 */
#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <vector>
#include <set>
#include <algorithm>

#include "./exec_pass.h"
#include "./graph_executor.h"
#include "../profiler/profiler.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"
#include "../operator/subgraph/subgraph_property.h"
#include "../operator/operator_common.h"

namespace mxnet {
namespace exec {

using namespace mxnet::common;

static const std::string GetDefaultSubgraphBackend() {
#if MXNET_USE_MKLDNN == 1
  return std::string("MKLDNN");
#else
  return std::string();
#endif
}

GraphExecutor::GraphExecutor(const nnvm::Symbol& symbol) {
  log_verbose_ = dmlc::GetEnv("MXNET_EXEC_VERBOSE_LOGGING", false);
  need_grad_ = false;
  is_dynamic_ = false;
  subgraph_property_ = dmlc::GetEnv("MXNET_SUBGRAPH_BACKEND", GetDefaultSubgraphBackend());
  if (subgraph_property_ == "NONE") {
    subgraph_property_ = std::string();
    LOG(INFO) << "MXNET_SUBGRAPH_BACKEND=NONE is detected, subgraph backend is not in use";
  }
  engine_ref_ = Engine::_GetSharedRef();
  symbol_ = symbol.Copy();
}

GraphExecutor::~GraphExecutor() {
  for (auto& n : op_nodes_) {
    if (n.cached_opr != nullptr) {
      Engine::Get()->DeleteOperator(n.cached_opr);
    }
  }
  // clean up seg ops
  for (auto& seg : cached_seg_opr_) {
    if (seg.opr != nullptr) {
      Engine::Get()->DeleteOperator(seg.opr);
    }
  }
}

void GraphExecutor::Forward(bool is_train) {
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  size_t sstep = static_cast<size_t>(step);
  if (sstep >= num_forward_nodes_) {
    *step_left = 0; return;
  }
  RunOps(is_train, sstep, sstep + 1);
  *step_left = static_cast<int>(num_forward_nodes_ - sstep - 1);
}

void GraphExecutor::Backward(const std::vector<NDArray>& head_grads, bool is_train) {
  {
    const auto& idx = graph_.indexed_graph();
    if (num_forward_inputs_ != idx.input_nodes().size()) {
      for (size_t i = 0; i < head_grad_array_.size(); ++i) {
        if (!head_grad_array_[i].is_none()) {
          CHECK(i < head_grads.size() && !head_grads[i].is_none())
              << "Because the last operator is not Loss function, "
              << "head_gradient is required when calling backward. "
              << "If you are attempting to minimize the output as "
              << "an objective, please modify your network and "
              << "pass it through the make_loss symbol.";
          const NDArray &from = head_grads[i];
          NDArray &to = head_grad_array_[i];
          if (this->is_dynamic_) {
            to.WaitToRead();
            if (!shape_is_known(to.shape())) {
              to.Init(from.shape());
            }
          }
          CopyFromTo(from, &to);
        }
      }
    }
  }
  if (this->is_dynamic_) {
    graph_ = InferShape(std::move(graph_), {}, "");
    mxnet::ShapeVector rshape = graph_.MoveCopyAttr<mxnet::ShapeVector>("shape");
    const auto& idx = graph_.indexed_graph();
    for (size_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) continue;
      OpNode& opnode = op_nodes_[nid];
      if (opnode.skip_exec_node) continue;
      for (NDArray &array : opnode.exec->in_array) {
        array.WaitToRead();
        if (!shape_is_known(array.shape())) {
          array.SetShapeFromChunk();
        }
      }
      int i = 0;
      for (NDArray &array : opnode.exec->in_array) {
        array.WaitToRead();
        if (!shape_is_known(array.shape())) {
          array.SetShapeFromChunk();
        }
        if (!shape_is_known(array.shape())) {
          mxnet::TShape shape = rshape[idx.entry_id(inode.inputs[i])];
          if (shape_is_known(shape)) {
            array.ReshapeAndAlloc(shape);
          }
        }
        ++i;
      }
      i = 0;
      for (NDArray &array : opnode.exec->out_array) {
        array.WaitToRead();
        if (!shape_is_known(array.shape())) {
          array.SetShapeFromChunk();
        }
        if (!shape_is_known(array.shape())) {
          mxnet::TShape shape = rshape[idx.entry_id(nid, i)];
          if (shape_is_known(shape)) {
            array.ReshapeAndAlloc(shape);
          }
        }
        ++i;
      }
    }
    graph_.attrs["shape"] = std::make_shared<dmlc::any>(rshape);
  }
  const auto& idx = graph_.indexed_graph();
  RunOps(is_train, num_forward_nodes_, idx.num_nodes());
}

void GraphExecutor::Print(std::ostream &os) const {  // NOLINT(*)
  nnvm::Symbol s; s.outputs = graph_.outputs;
  s.Print(os);
  // message to be backward compatible with the memonger
  size_t total_bytes = graph_.GetAttr<size_t>("storage_allocated_bytes");
  os << "Total " << (total_bytes >> 20UL) <<" MB allocated\n";
  os << "Total " << 11 << " TempSpace resource requested\n";
}

/*!
 * \brief Return the "optimized" symbol contained in the executor graph.
 */
nnvm::Symbol GraphExecutor::GetOptimizedSymbol() {
  Symbol ret;
  ret.outputs = std::vector<nnvm::NodeEntry>(graph_.outputs.begin(),
      graph_.outputs.begin() + num_forward_outputs_);
  return ret.Copy();
}

void GraphExecutor::SetMonitorCallback(const MonitorCallback& callback, bool monitor_all) {
  CHECK(callback) << "invalid callback";
  monitor_callback_ = callback;
  monitor_all_ = monitor_all;
}

const std::vector<NDArray>& GraphExecutor::outputs() const {
  if (this->is_dynamic_) {
    for (const NDArray &array : output_arrays_) {
      array.WaitToRead();
      if (!shape_is_known(array.shape())) {
        const_cast<NDArray &>(array).SetShapeFromChunk();
      }
    }
  }
  return output_arrays_;
}

const std::unordered_map<std::string, NDArray>& GraphExecutor::in_arg_map() const {
  return in_arg_map_;
}

const std::unordered_map<std::string, NDArray>& GraphExecutor::arg_grad_map() const {
  return arg_grad_map_;
}

const std::unordered_map<std::string, NDArray>& GraphExecutor::aux_state_map() const {
  return aux_state_map_;
}

static nnvm::NodeEntry AttrHint(nnvm::NodeEntry src, nnvm::NodeEntry like) {
  static const Op* id_like = Op::Get("_identity_with_attr_like_rhs");
  nnvm::ObjectPtr n = nnvm::Node::Create();
  n->attrs.op = id_like;
  n->attrs.name = src.node->attrs.name + "_id";
  n->inputs = {src, like};
  return nnvm::NodeEntry{n, 0, 0};
}

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v) {
  using nnvm::Op;
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  static const Op* ewise_sum_op = Op::Get("ElementWiseSum");
  static const Op* identity_op = Op::Get("identity");
  static const Op* zeros_op = Op::Get("_zeros");
  static const Op* zeros_like_op = Op::Get("zeros_like");

  if (v.empty()) {
    nnvm::ObjectPtr ng = nnvm::Node::Create();
    ng->attrs.op = Op::Get("_zeros_without_dtype");
    ng->attrs.name = "zeros_without_dtype";
    ng->attrs.op->attr_parser(&(ng->attrs));
    return nnvm::NodeEntry(std::move(ng), 0, 0);
  }

  // remove zero in the sum. at least keep 1.
  auto begin = std::remove_if(v.begin(), v.end(), [](const nnvm::NodeEntry& nodeEntry) {
     CHECK(nodeEntry.node);
     return nodeEntry.node->op() == zeros_op || nodeEntry.node->op() == zeros_like_op;
  });
  if (begin == v.begin()) ++begin;
  v.erase(begin, v.end());
  CHECK(!v.empty());

  if (v.size() == 1) {
    return std::move(v[0]);
  } else {
    if (v.size() < inplace_sum_cap) {
      nnvm::ObjectPtr sum_node = nnvm::Node::Create();
      sum_node->attrs.op = ewise_sum_op;
      sum_node->attrs.name = "sum_grad";
      sum_node->attrs.dict["num_args"] = std::to_string(v.size());
      sum_node->attrs.op->attr_parser(&(sum_node->attrs));
      sum_node->inputs = std::move(v);
      return nnvm::NodeEntry(std::move(sum_node), 0, 0);
    } else {
      // use a stream line of plus instead
      nnvm::NodeEntry ret = v[0];
      for (size_t i = 1; i < v.size(); ++i) {
        // Add control flow dependency from to previous node
        // This enforces the gradient sum order will be in the inverse
        // order of forward traversal
        // NOTE: adding control dependency can be dangerous and cause cycle in the dep.
        // The curent usage is correct, because of the following invariant:
        // assert: v[i-1] do not depend on v[i]
        // To put in plain text: v is gradient vector that get pushed in the order
        // that can generate them, which means if v[i] is not yet pushed,
        // all previous gradient cannot depend on it.
        // Note: For a symbol like the following:
        // data = mx.sym.Variable('data')
        // sym = data + data + data + data + data + data + data
        // the node entries v passed in here are of the same node of
        // op _identity_with_attr_like_rhs. We should skip adding a node
        // to its own control_deps.
        if (v[i-1].node != v[i].node) {
          v[i].node->control_deps.push_back(ret.node);
        }

        std::ostringstream os;
        os << "sum_grad_" << i;
        nnvm::ObjectPtr x = nnvm::Node::Create();
        x->attrs.op = ewise_plus_op;
        x->attrs.name = os.str();
        x->inputs = {ret, v[i]};
        ret = nnvm::NodeEntry(std::move(x), 0, 0);
      }
      // identity node is used to avoid exposure of dummy plus node
      // when its output get assigned to another space.
      nnvm::ObjectPtr id_node = nnvm::Node::Create();
      id_node->attrs.op = identity_op;
      id_node->attrs.name = "sum_grad_final";
      id_node->inputs = {ret};
      return nnvm::NodeEntry{id_node, 0, 0};
    }
  }
}

template<typename ValueType>
inline ValueType get_node_attr(
    const nnvm::Node& node,
    const std::string& key, ValueType default_value) {
  auto it = node.attrs.dict.find(key);
  if (it == node.attrs.dict.end()) {
    return default_value;
  } else {
    ValueType ret;
    dmlc::parameter::FieldEntry<ValueType> e;
    e.Init(key, &ret, ret);
    e.Set(&ret, it->second);
    return ret;
  }
}

/*!
 * \brief Create the graph for backward pass.
 * This is triggered by both simple_bind and bind flows.
 */
nnvm::Graph GraphExecutor::InitFullGraph(nnvm::Symbol symbol,
                                         const std::vector<OpReqType>& grad_req_types) {
  using nnvm::ObjectPtr;
  using nnvm::NodeEntry;
  // initial information
  num_forward_outputs_ = symbol.outputs.size();
  num_forward_inputs_ = symbol.ListInputs(nnvm::Symbol::kAll).size();

  nnvm::Graph g;
  g.outputs = symbol.outputs;
  bool do_elim_common_expr = dmlc::GetEnv("MXNET_ELIMINATE_COMMON_EXPR", true);
  if (do_elim_common_expr)
    g = exec::EliminateCommonExpr(std::move(g));
  need_grad_ = false;
  for (OpReqType req : grad_req_types) {
    if (req != kNullOp) need_grad_ = true;
  }
  if (!need_grad_) return g;
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    NodeEntry ngrad(nnvm::Node::Create(), 0, 0);
    ngrad.node->attrs.name = "_head_grad_" + std::to_string(i);
    head_grad_entry_.emplace_back(AttrHint(ngrad, g.outputs[i]));
    head_grad_map_[ngrad.node.get()] = i;
  }
  std::vector<ObjectPtr> args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);
  std::vector<NodeEntry> xs;
  for (size_t i = 0; i < grad_req_types.size(); ++i) {
    if (grad_req_types[i] != kNullOp) {
      xs.emplace_back(args[i]);
    }
  }

  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  auto need_mirror = [do_mirror](const nnvm::Node& node) -> int {
    if (node.is_variable()) return 0;
    const std::string& type = node.attrs.op->name;
    if (type == "Dropout") return false;
    if (get_node_attr(node, "__force_mirroring__", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    return true;
  };

  std::vector<const nnvm::Op*> zero_ops;
  zero_ops.push_back(nnvm::Op::Get("zeros_like"));
  zero_ops.push_back(nnvm::Op::Get("_zeros"));

  // take gradient
  nnvm::Graph g_grad = nnvm::pass::MXGradient(
      g, symbol.outputs, xs, head_grad_entry_,
      AggregateGradient, need_mirror, nullptr,
      zero_ops, "_copy");
  CHECK_EQ(g_grad.outputs.size(), xs.size());
  for (const auto &e : g_grad.outputs) {
    g.outputs.push_back(e);
  }

  return g;
}

/*!
 * \brief GraphExecutor initializer for regular bind flow in which
 * input arguments and gradients are provided by users. This initializer
 * uses the user provided NDArrays to populate data entries of the graph.
 */
void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<NDArray>& in_args,
                         const std::vector<NDArray>& arg_grad_store,
                         const std::vector<OpReqType>& grad_req_types,
                         const std::vector<NDArray>& aux_states,
                         Executor* shared_exec,
                         const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  // create in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes
  auto get_ctx1 = [](const NDArray& nd) { return nd.ctx(); };
  auto get_ctx2 = [default_ctx](const NDArray& nd) -> Context {
    if (nd.is_none()) return default_ctx;
    return nd.ctx();
  };
  std::vector<Context> in_arg_ctxes(in_args.size());
  std::transform(in_args.begin(), in_args.end(), in_arg_ctxes.begin(), get_ctx1);
  std::vector<Context> arg_grad_ctxes(arg_grad_store.size());
  std::transform(arg_grad_store.begin(), arg_grad_store.end(), arg_grad_ctxes.begin(), get_ctx2);
  std::vector<Context> aux_state_ctxes(aux_states.size());
  std::transform(aux_states.begin(), aux_states.end(), aux_state_ctxes.begin(), get_ctx1);

  nnvm::Graph g = InitGraph(symbol, default_ctx, ctx_map, in_arg_ctxes,
                            arg_grad_ctxes, aux_state_ctxes, grad_req_types);

  // create arg_shapes and arg_dtypes for shape and type inferences
  const auto& idx = g.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  size_t arg_top = 0, aux_top = 0;
  data_entry_.resize(idx.num_node_entries());
  mxnet::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_dtypes;
  StorageTypeVector arg_stypes(idx.num_node_entries(), -1);
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& arg_name = idx[nid].source->attrs.name;
    size_t eid = idx.entry_id(nid, 0);
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      data_entry_[eid] = aux_states[aux_top];
      arg_shapes.push_back(aux_states[aux_top].shape());
      arg_dtypes.push_back(aux_states[aux_top].dtype());
      arg_stypes[eid] = aux_states[aux_top].storage_type();
      aux_state_map_.emplace(arg_name, aux_states[aux_top]);
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      data_entry_[eid] = in_args[arg_top];
      arg_shapes.push_back(in_args[arg_top].shape());
      arg_dtypes.push_back(in_args[arg_top].dtype());
      arg_stypes[eid] = in_args[arg_top].storage_type();
      in_arg_map_.emplace(arg_name, in_args[arg_top]);
      if (kNullOp != grad_req_types[arg_top]) {
        auto grad_oid = grad_store_.size() + num_forward_outputs_;
        auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
        arg_stypes[grad_eid] = arg_grad_store[arg_top].storage_type();
        grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_store[arg_top]);
        arg_grad_map_.emplace(arg_name, arg_grad_store[arg_top]);
        if (log_verbose_) {
          LOG(INFO) << "\tassign data entry\t" << grad_eid << " as "
                    << common::stype_string(arg_stypes[grad_eid]) << " (grad)";
        }
      }
      ++arg_top;
    }
    if (log_verbose_) {
      LOG(INFO) << "\tassign data entry\t" << eid << " as "
                << common::stype_string(data_entry_[eid].storage_type()) << " (input)";
    }
  }

  // expand arg_shapes and arg_dtypes to contain backward inputs
  arg_shapes.resize(idx.input_nodes().size(), mxnet::TShape());
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    this->is_dynamic_ = true;
  }

  arg_dtypes.resize(idx.input_nodes().size(), -1);
  g = InferType(std::move(g), std::move(arg_dtypes), "__dtype__");
  if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) {
    HandleInferTypeError(num_forward_inputs_, g.indexed_graph(),
                         g.GetAttr<nnvm::DTypeVector>("dtype"));
  }

  g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(arg_stypes));
  g = InferStorageType(std::move(g), StorageTypeVector(), "");
  if (g.GetAttr<size_t>("storage_type_num_unknown_nodes") != 0U) {
    HandleInferStorageTypeError(num_forward_inputs_, g.indexed_graph(),
                                g.GetAttr<StorageTypeVector>("storage_type"));
  }

  // Initialize the rest attributes of the graph.
  // This function can be called by regular bind
  // operation flow as well.
  FinishInitGraph(symbol, g, shared_exec, feed_dict);
}

/*!
 * \brief Initialize in_args, arg_grads, and aux_states
 * and their data_entry_ of the executor. This function
 * is called for regular simple_bind flow, i.e. no
 * shared data arrays are provided.
 */
void GraphExecutor::InitArguments(const nnvm::IndexedGraph& idx,
                                  const mxnet::ShapeVector& inferred_shapes,
                                  const nnvm::DTypeVector& inferred_dtypes,
                                  const StorageTypeVector& inferred_stypes,
                                  const std::vector<Context>& in_arg_ctxes,
                                  const std::vector<Context>& arg_grad_ctxes,
                                  const std::vector<Context>& aux_state_ctxes,
                                  const std::vector<OpReqType>& grad_req_types,
                                  std::vector<NDArray>* in_arg_vec,
                                  std::vector<NDArray>* arg_grad_vec,
                                  std::vector<NDArray>* aux_state_vec) {
  // initialize in_args, arg_grads, and aux_states
  // populate grad_store_
  data_entry_.resize(idx.num_node_entries());
  size_t arg_top = 0, aux_top = 0;
  const auto& mutable_nodes = idx.mutable_input_nodes();
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const mxnet::TShape& inferred_shape = inferred_shapes[eid];
    const int inferred_dtype = inferred_dtypes[eid];
    const NDArrayStorageType inferred_stype = (NDArrayStorageType) inferred_stypes[eid];
    const std::string& arg_name = idx[nid].source->attrs.name;
    if (mutable_nodes.count(nid)) {  // aux_states
      EmplaceBackZeros(inferred_stype, inferred_shape, aux_state_ctxes[aux_top],
                       inferred_dtype, aux_state_vec);
      data_entry_[eid] = aux_state_vec->back();
      aux_state_map_.emplace(arg_name, aux_state_vec->back());
      ++aux_top;
      if (log_verbose_) {
        LOG(INFO) << "\tassign aux entry\t" << eid << "\t as "
                  << common::stype_string(inferred_stype);
      }
    } else {  // in_args
      EmplaceBackZeros(inferred_stype, inferred_shape, in_arg_ctxes[arg_top],
                       inferred_dtype, in_arg_vec);
      data_entry_[eid] = in_arg_vec->back();
      if (log_verbose_) {
        LOG(INFO) << "\tassign data entry\t" << eid << "\tas "
                  << common::stype_string(inferred_stype);
      }
      // Get the storage type for grad
      if (kNullOp == grad_req_types[arg_top]) {
        arg_grad_vec->emplace_back();
      } else {
        // Init based on storage type
        auto grad_oid = grad_store_.size() + num_forward_outputs_;
        auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
        auto grad_stype = (NDArrayStorageType) inferred_stypes[grad_eid];
        EmplaceBackZeros(grad_stype, inferred_shape, arg_grad_ctxes[arg_top],
                         inferred_dtype, arg_grad_vec);
        if (log_verbose_) {
          LOG(INFO) << "\tassign grad entry\t" << grad_eid << "\tas "
                    << common::stype_string(grad_stype);
        }
        grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_vec->back());
        arg_grad_map_.emplace(arg_name, arg_grad_vec->back());
      }
      in_arg_map_.emplace(arg_name, in_arg_vec->back());
      ++arg_top;
    }
  }
}

/*!
 * \brief Initialize in_args, arg_grads, and aux_states
 * and their data_entry_ of the executor using
 * shared_buffer from DataParallelExecutorGroup
 * and shared_exec if available.
 */
void GraphExecutor::InitArguments(const nnvm::IndexedGraph& idx,
                                  const mxnet::ShapeVector& inferred_shapes,
                                  const nnvm::DTypeVector& inferred_dtypes,
                                  const StorageTypeVector& inferred_stypes,
                                  const std::vector<Context>& in_arg_ctxes,
                                  const std::vector<Context>& arg_grad_ctxes,
                                  const std::vector<Context>& aux_state_ctxes,
                                  const std::vector<OpReqType>& grad_req_types,
                                  const std::unordered_set<std::string>& shared_arg_names,
                                  const Executor* shared_exec,
                                  std::unordered_map<std::string, NDArray>* shared_buffer,
                                  std::vector<NDArray>* in_arg_vec,
                                  std::vector<NDArray>* arg_grad_vec,
                                  std::vector<NDArray>* aux_state_vec) {
  // initialize in_args, arg_grads, and aux_states and populate grad_store_
  data_entry_.resize(idx.num_node_entries());
  size_t arg_top = 0, aux_top = 0;
  const auto& mutable_nodes = idx.mutable_input_nodes();
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const mxnet::TShape& inferred_shape = inferred_shapes[eid];
    const int inferred_dtype = inferred_dtypes[eid];
    const NDArrayStorageType inferred_stype = (NDArrayStorageType) inferred_stypes[eid];
    const std::string& arg_name = idx[nid].source->attrs.name;
    // aux_states
    if (mutable_nodes.count(nid)) {
      if (nullptr != shared_exec) {
        const NDArray& aux_nd = shared_exec->aux_state_map().at(arg_name);
        CHECK(inferred_stype == kDefaultStorage && aux_nd.storage_type() == kDefaultStorage)
          << "Non-default storage type detected when creating auxilliary NDArray. The allocated "
          << "memory of shared_exec.aux_array cannot be resued for argument: "
          << arg_name << " for the current executor";
        CHECK_EQ(inferred_shape, aux_nd.shape())
          << "Inferred shape does not match shared_exec.aux_array's shape."
             " Therefore, the allocated memory for shared_exec.aux_array cannot"
             " be resued for creating auxilliary NDArray of the argument: "
          << arg_name << " for the current executor";
        CHECK_EQ(inferred_dtype, aux_nd.dtype())
          << "Inferred dtype does not match shared_exec.aux_array's dtype."
             " Therefore, the allocated memory for shared_exec.aux_array cannot"
             " be resued for creating auxilliary NDArray of the argument: "
          << arg_name << " for the current executor";
        aux_state_vec->emplace_back(aux_nd);
      } else {
        EmplaceBackZeros(inferred_stype, inferred_shape, aux_state_ctxes[aux_top],
                         inferred_dtype, aux_state_vec);
      }  // if (has_shared_exec)
      data_entry_[eid] = aux_state_vec->back();
      aux_state_map_.emplace(arg_name, aux_state_vec->back());
      ++aux_top;
    } else {  // in_args and grad for in_args
      if (shared_arg_names.count(arg_name)) {  // model parameter
        // model parameter
        if (nullptr != shared_exec) {
          const NDArray& in_arg_nd = shared_exec->in_arg_map().at(arg_name);
          auto arg_nd_stype = in_arg_nd.storage_type();
          // for model parameter, both default storage and row_sparse storage can be shared
          bool shareable_arg_stype = inferred_stype == kDefaultStorage ||
                                     inferred_stype == kRowSparseStorage;
          // try to reuse memory from shared_exec
          CHECK(shareable_arg_stype) << "Inferred storage type "
            << common::stype_string(inferred_stype)
            << " does not support memory sharing with shared_exec.arg_array";
          CHECK_EQ(inferred_stype, arg_nd_stype)
            << "Inferred stype does not match shared_exec.arg_array's stype"
               " Therefore, the allocated memory for shared_exec.arg_array cannot"
               " be resued for creating NDArray of the argument "
            << arg_name << " for the current executor";
          CHECK_EQ(inferred_shape, in_arg_nd.shape())
            << "Inferred shape does not match shared_exec.arg_array's shape"
               " Therefore, the allocated memory for shared_exec.arg_array cannot"
               " be resued for creating NDArray of the argument "
            << arg_name << " for the current executor";
          CHECK_EQ(inferred_dtype, in_arg_nd.dtype())
            << "Inferred dtype does not match shared_exec.arg_array's dtype"
               " Therefore, the allocated memory for shared_exec.arg_array cannot"
               " be resued for creating NDArray of the argument "
            << arg_name << " for the current executor";
          in_arg_vec->emplace_back(in_arg_nd);
        } else {
          // doesn't have shared_exec, or non-default storage
          EmplaceBackZeros(inferred_stype, inferred_shape, in_arg_ctxes[arg_top],
                           inferred_dtype, in_arg_vec);
        }
        // gradient for model parameter
        if (kNullOp == grad_req_types[arg_top]) {
          arg_grad_vec->emplace_back();
        } else {
          auto grad_oid = grad_store_.size() + num_forward_outputs_;
          auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
          auto grad_stype = (NDArrayStorageType) inferred_stypes[grad_eid];
          if (nullptr != shared_exec && grad_stype == kDefaultStorage &&
              shared_exec->arg_grad_map().at(arg_name).storage_type() == kDefaultStorage) {
            // try to reuse memory from shared_exec
            arg_grad_vec->emplace_back(shared_exec->arg_grad_map().at(arg_name));
          } else {
            // no need to reuse memory from shared_exec for gradient of non-default storage
            EmplaceBackZeros(grad_stype, inferred_shape, arg_grad_ctxes[arg_top],
                             inferred_dtype, arg_grad_vec);
          }
          grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_vec->back());
        }
      } else {  // !shared_arg_names.count(arg_name)
        // model parameter, row_sparse ndarray sharing enabled
        bool enable_row_sparse_sharing = true;
        in_arg_vec->emplace_back(ReshapeOrCreate(arg_name, inferred_shape, inferred_dtype,
                                                 inferred_stype, in_arg_ctxes[arg_top],
                                                 shared_buffer, enable_row_sparse_sharing));
        // gradient for model parameter, row_sparse ndarray sharing disabled
        if (kNullOp == grad_req_types[arg_top]) {
          arg_grad_vec->emplace_back();
        } else {
          auto grad_oid = grad_store_.size() + num_forward_outputs_;
          auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
          auto grad_stype = (NDArrayStorageType) inferred_stypes[grad_eid];
          bool enable_row_sparse_sharing = false;
          arg_grad_vec->emplace_back(ReshapeOrCreate("grad of " + arg_name, inferred_shape,
                                                     inferred_dtype, grad_stype,
                                                     arg_grad_ctxes[arg_top], shared_buffer,
                                                     enable_row_sparse_sharing));
          grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_vec->back());
        }  // if (kNullOp == grad_req_types[arg_top])
      }  // if (shared_arg_names.count(arg_name))
      in_arg_map_.emplace(arg_name, in_arg_vec->back());
      if (!arg_grad_vec->back().is_none()) {
        arg_grad_map_.emplace(arg_name, arg_grad_vec->back());
      }
      data_entry_[eid] = in_arg_vec->back();
      ++arg_top;
    }
  }
}

/*!
 * \brief Finish graph initialization after shape and dtype inferences.
 * This function is used by both simple_bind and bind flows.
 */
void GraphExecutor::FinishInitGraph(nnvm::Symbol symbol,
                                    nnvm::Graph g,
                                    Executor* shared_exec,
                                    const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  const auto& idx = g.indexed_graph();
  const auto& vstorage_type = g.GetAttr<StorageTypeVector>("storage_type");

  // data entries for output gradients
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    data_entry_[idx.entry_id(idx.outputs()[j])] = grad_store_[j - num_forward_outputs_].second;
  }

  {
    // memory allocator
    nnvm::StorageVector arg_storage_id(idx.num_node_entries(), kBadStorageID);
    for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
      arg_storage_id[idx.entry_id(idx.outputs()[j])] = kExternalStorageID;
    }
    for (const auto& kv : feed_dict) {
      uint32_t eid = idx.entry_id(kv.first);
      data_entry_[eid] = kv.second;
      arg_storage_id[eid] = kExternalStorageID;
    }
    for (size_t i = 0; i < idx.num_node_entries(); i++) {
      if (vstorage_type[i] != kDefaultStorage) arg_storage_id[i] = kDynamicStorageID;
    }
    g.attrs["storage"] = std::make_shared<dmlc::any>(std::move(arg_storage_id));
    g = nnvm::ApplyPass(g, "MXPlanMemory");
  }
  g = DetectInplaceAddTo(g);

  // log the static memory plan of the graph
  static bool mem_log_verbose = dmlc::GetEnv("MXNET_MEM_PLAN_VERBOSE_LOGGING", false);
  if (mem_log_verbose) {
    common::LogMemoryPlan(g);
  }

  g = AttachOpExecs(g);
  AttachOpResources(g);
  graph_ = std::move(g);

  if (shared_exec != nullptr) {
    this->InitDataEntryMemory(&(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_));
  } else {
    this->InitDataEntryMemory(nullptr);
  }

  {
    // initialize output arrays
    auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < num_forward_outputs_; ++i) {
      auto& e = idx.outputs()[i];
      output_arrays_.push_back(data_entry_[idx.entry_id(e)]);
    }
    // initialize head gradient array
    head_grad_array_.resize(symbol.outputs.size());
    for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes().at(i);
      uint32_t oid = head_grad_map_.at(idx[nid].source);
      head_grad_array_[oid] = data_entry_[idx.entry_id(nid, 0)];
    }
  }
  this->InitCachedOps();
  this->InitOpSegs();
}

/*!
 * \brief GraphExecutor initializer for simple bind flow in
 * which only certain input shapes and dtypes are provided by users.
 * The initializer uses these shapes and dtypes to perform
 * shape and dtype inferences, and then create NDArrays
 * to populate data entries of the graph. The created NDArrays
 * for in_args, arg_grads and aux_states are passed to the
 * front end to attach the created executor.
 * In front end, if the simple_bind flow is trigger by
 * _bind_ith_exec, the shared data arrays of DataParallelExecutorGroup
 * and shared executor will be taken into account in creating
 * NDArrays for in_args, arg_grads, and aux_states for resuing
 * already allocated memory.
 */
void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<Context>& in_arg_ctxes,
                         const std::vector<Context>& arg_grad_ctxes,
                         const std::vector<Context>& aux_state_ctxes,
                         const std::unordered_map<std::string, mxnet::TShape>& arg_shape_map,
                         const std::unordered_map<std::string, int>& arg_dtype_map,
                         const std::unordered_map<std::string, int>& arg_stype_map,
                         const std::vector<OpReqType>& grad_req_types,
                         const std::unordered_set<std::string>& shared_arg_names,
                         std::vector<NDArray>* in_arg_vec,
                         std::vector<NDArray>* arg_grad_vec,
                         std::vector<NDArray>* aux_state_vec,
                         std::unordered_map<std::string, NDArray>* shared_buffer,
                         Executor* shared_exec,
                         const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  nnvm::Graph g = InitGraph(symbol, default_ctx, ctx_map, in_arg_ctxes, arg_grad_ctxes,
                            aux_state_ctxes, grad_req_types);

  // The following code of shape and dtype inferences and argument
  // initialization is for simple_bind only. Regular bind operation
  // should do this differently.

  // Initialize arg_shapes and arg_dtypes for shape and type inferences.
  // It contains all in_args and aux_states' shapes and types in a certain order.
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  mxnet::ShapeVector arg_shapes(idx.input_nodes().size(), mxnet::TShape());
  nnvm::DTypeVector arg_dtypes(idx.input_nodes().size(), -1);
  StorageTypeVector arg_stypes(idx.input_nodes().size(), kUndefinedStorage);
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    auto it1 = arg_shape_map.find(name);
    if (arg_shape_map.end() != it1) {
      arg_shapes[i] = it1->second;
    }
    auto it2 = arg_dtype_map.find(name);
    if (arg_dtype_map.end() != it2) {
      arg_dtypes[i] = it2->second;
    }
    auto it3 = arg_stype_map.find(name);
    if (arg_stype_map.end() != it3) {
      arg_stypes[i] = it3->second;
    }
  }
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    HandleInferShapeError(num_forward_inputs_, g.indexed_graph(),
                          g.GetAttr<mxnet::ShapeVector>("shape"));
  }

  g = InferType(std::move(g), std::move(arg_dtypes), "__dtype__");
  if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) {
    HandleInferTypeError(num_forward_inputs_, g.indexed_graph(),
                         g.GetAttr<nnvm::DTypeVector>("dtype"));
  }

  g = InferStorageType(std::move(g), std::move(arg_stypes), "__storage_type__");
  if (g.GetAttr<size_t>("storage_type_num_unknown_nodes") != 0U) {
    HandleInferStorageTypeError(num_forward_inputs_, g.indexed_graph(),
                                g.GetAttr<StorageTypeVector>("storage_type"));
  }

  // Create in_args, arg_grads, and aux_states using
  // the inferred shapes and dtypes.
  if (nullptr == shared_buffer) {  // regular simple bind
    InitArguments(idx, g.GetAttr<mxnet::ShapeVector>("shape"),
                  g.GetAttr<nnvm::DTypeVector>("dtype"),
                  g.GetAttr<StorageTypeVector>("storage_type"),
                  in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,
                  grad_req_types, in_arg_vec, arg_grad_vec, aux_state_vec);
  } else {  // simple bind using shared data arrays and shared_exec
    InitArguments(idx, g.GetAttr<mxnet::ShapeVector>("shape"),
                  g.GetAttr<nnvm::DTypeVector>("dtype"),
                  g.GetAttr<StorageTypeVector>("storage_type"),
                  in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,
                  grad_req_types, shared_arg_names, shared_exec,
                  shared_buffer, in_arg_vec, arg_grad_vec, aux_state_vec);
  }
  // The above code of shape and dtype inferences and argument
  // initialization is for simple_bind only. Regular bind operation
  // should do this differently.

  // Initialize the rest attributes of the graph.
  // This function can be called by regular bind
  // operation flow as well.
  FinishInitGraph(symbol, g, shared_exec, feed_dict);
}

/*!
 * \brief Return a new executor with the same symbol and shared memory,
 * but different input/output shapes.
 * For runtime reshaping, variable length sequences, etc.
 * The returned executor shares state with the current one,
 * and cannot be used in parallel with it.
 */
Executor* GraphExecutor::Reshape(const bool partial_shaping,
                                 const bool allow_up_sizing,
                                 const Context& default_ctx,
                                 const std::map<std::string, Context>& ctx_map,
                                 const std::unordered_map<std::string, mxnet::TShape>&
                                   provided_arg_shapes,
                                 std::vector<NDArray>* in_args,
                                 std::vector<NDArray>* arg_grads,
                                 std::vector<NDArray>* aux_states) {
  nnvm::Graph g;
  nnvm::Symbol symbol;
  symbol.outputs = symbol_.outputs;
  g.outputs = symbol_.outputs;
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  mxnet::ShapeVector arg_shapes(idx.input_nodes().size(), mxnet::TShape());
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    auto it = provided_arg_shapes.find(name);
    if (provided_arg_shapes.end() != it) {
      arg_shapes[i] = it->second;
    }
  }
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    this->is_dynamic_ = true;
  }
  const mxnet::ShapeVector& shape_vec = g.GetAttr<mxnet::ShapeVector>("shape");
  std::vector<OpReqType> grad_req_types;
  size_t grad_top = 0;
  const size_t num_args = in_arg_map_.size();
  const size_t num_aux = aux_state_map_.size();
  in_args->reserve(num_args);
  grad_req_types.reserve(num_args);
  arg_grads->reserve(num_args);
  aux_states->reserve(num_aux);
  for (uint32_t nid : idx.input_nodes()) {
    std::string name = idx[nid].source->attrs.name;
    const mxnet::TShape& new_shape = shape_vec[idx.entry_id(nid, 0)];
    if (idx.mutable_input_nodes().count(nid) == 0) {
      NDArray& arr = in_arg_map_.at(name);
      auto it = arg_grad_map_.find(name);
      if (partial_shaping || provided_arg_shapes.count(name) || new_shape == arr.shape()) {
        if (new_shape.Size() > arr.shape().Size()) {
          CHECK(allow_up_sizing) << "New shape of arg: " << name << " is larger than original."
            << "First making a big executor and then down sizing it "
            << "is more efficient than the reverse."
            << "If you really want to up size, set allow_up_sizing=True "
            << "to enable allocation of new arrays.";
          in_args->emplace_back(new_shape, arr.ctx(), false, arr.dtype());
          if (it != arg_grad_map_.end()) {
            NDArray& darr = it->second;
            arg_grads->emplace_back(new_shape, darr.ctx(), false, darr.dtype());
            grad_req_types.push_back(grad_store_.at(grad_top++).first);
          } else {
            arg_grads->emplace_back();
            grad_req_types.push_back(kNullOp);
          }
        } else {
          in_args->push_back(arr.Reshape(new_shape));
          if (it != arg_grad_map_.end()) {
            NDArray& darr = it->second;
            arg_grads->push_back(darr.Reshape(new_shape));
            grad_req_types.push_back(grad_store_.at(grad_top++).first);
          } else {
            arg_grads->emplace_back();
            grad_req_types.push_back(kNullOp);
          }
        }
      } else {
        LOG(FATAL) << "Shape of unspecifie arg: " << name << " changed. "
          << "This can cause the new executor to not share parameters "
          << "with the old one. Please check for error in network."
          << "If this is intended, set partial_shaping=True to suppress this warning.";
      }
    } else {
      NDArray& arr = aux_state_map_.at(name);
      if (partial_shaping || new_shape == arr.shape()) {
        if (new_shape.Size() > arr.shape().Size()) {
          CHECK(allow_up_sizing) << "New shape of arg: " << name << " is larger than original."
            << "First making a big executor and then down sizing it "
            << "is more efficient than the reverse."
            << "If you really want to up size, set allow_up_sizing=True "
            << "to enable allocation of new arrays.";
          aux_states->emplace_back(new_shape, arr.ctx(), false, arr.dtype());
        } else {
          aux_states->push_back(arr.Reshape(new_shape));
        }
      } else {
        LOG(FATAL) << "Shape of unspecifie arg: " << name << " changed. "
          << "This can cause the new executor to not share parameters "
          << "with the old one. Please check for error in network."
          << "If this is intended, set partial_shaping=True to suppress this warning.";
      }
    }
  }
  auto exec = new GraphExecutor(symbol);
  exec->Init(symbol.Copy(), default_ctx, ctx_map,
             *in_args, *arg_grads, grad_req_types, *aux_states,
             this);
  return exec;
}

/*!
 * \brief This function is triggered by both simple_bind
 * and bind flows.
 * Setup backward graph, create device and context
 * attributes in the graph, and calculate the number
 * of forward nodes.
 */
Graph GraphExecutor::InitGraph(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& ctx_map,
                               const std::vector<Context>& in_arg_ctxes,
                               const std::vector<Context>& arg_grad_ctxes,
                               const std::vector<Context>& aux_state_ctxes,
                               const std::vector<OpReqType>& grad_req_types) {
  // setup gradient
  nnvm::Graph g = InitFullGraph(symbol, grad_req_types);

#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC && !defined(_WIN32)
  if (default_ctx.dev_mask() == Context::kGPU && dmlc::GetEnv("MXNET_USE_FUSION", true)) {
    nnvm::Graph unoptimized_graph;
    common::CopyGraph(&unoptimized_graph, g, false);

    if (common::CheckForInputNameDuplicates(unoptimized_graph.indexed_graph())) {
      g.attrs["num_forward_outputs"] = std::make_shared<nnvm::any>(num_forward_outputs_);
      g = FusePointwiseForward(std::move(g));
      g.attrs["num_forward_outputs"] = std::make_shared<nnvm::any>(num_forward_outputs_);
      g = FusePointwiseBackward(std::move(g));
      // Check the topological order of inputs
      const auto &original_inputs = unoptimized_graph.indexed_graph().input_nodes();
      const auto &new_inputs = g.indexed_graph().input_nodes();
      if (original_inputs.size() != new_inputs.size()) {
        LOG(WARNING)
          << "Number of inputs after fusion does not match original number of inputs. "
          << "This is most probably a bug. Disabling fusion for this run.";
        g = unoptimized_graph;
      } else {
        for (size_t i = 0; i < new_inputs.size(); ++i) {
          if (unoptimized_graph.indexed_graph()[original_inputs[i]].source->attrs.name !=
              g.indexed_graph()[new_inputs[i]].source->attrs.name) {
            LOG(WARNING) << "Disabling fusion due to altered topological order of inputs.";
            g = unoptimized_graph;
            break;
          }
        }
      }
    } else {
      LOG(WARNING)
        << "Graph contains duplicate names for some of its inputs - fusion is NOT enabled!";
     }
  }
#else
  // Only warn user if MXNET_USE_FUSION env var is explicitly set
  if (default_ctx.dev_mask() == Context::kGPU && dmlc::GetEnv("MXNET_USE_FUSION", false)) {
    WarnFusionNotSupported();
  }
#endif  // MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC && !defined(_WIN32)

  // create "device" and "context" attrs for the graph
  g = AssignContext(g, default_ctx, ctx_map,
                    in_arg_ctxes,
                    arg_grad_ctxes,
                    aux_state_ctxes,
                    grad_req_types,
                    num_forward_inputs_,
                    num_forward_outputs_);

  const auto& idx = g.indexed_graph();
  // get number of nodes used in forward pass
  num_forward_nodes_ = 0;
  for (size_t i = 0; i < num_forward_outputs_; ++i) {
    num_forward_nodes_ = std::max(
        num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  return g;
}

// initialize the memory of each entries
void GraphExecutor::InitDataEntryMemory(std::vector<NDArray>* shared_pool) {
  using nnvm::DTypeVector;
  using mxnet::ShapeVector;
  using nnvm::StorageVector;
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // get the storage
  const auto& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const auto& vshape = graph_.GetAttr<mxnet::ShapeVector>("shape");
  const auto& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  const auto& vstorage_type = graph_.GetAttr<StorageTypeVector>("storage_type");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  CHECK_EQ(idx.num_node_entries(), vshape.size());
  CHECK_EQ(idx.num_node_entries(), vdtype.size());
  CHECK_EQ(idx.num_node_entries(), vstorage.size());
  CHECK_EQ(data_entry_.size(), vshape.size());
  std::vector<Context> data_context(idx.num_node_entries());
  std::vector<NDArrayStorageType> data_storage_type(idx.num_node_entries(), kUndefinedStorage);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    for (uint32_t i = 0; i < idx[nid].source->num_outputs(); ++i) {
      auto eid = idx.entry_id(nid, i);
      data_context[eid] = vctx[nid];
      CHECK_NE(vstorage_type[eid], kUndefinedStorage);
      data_storage_type[eid] = (NDArrayStorageType) vstorage_type[eid];
    }
  }

  // information about the pool
  struct PoolEntry {
    Context ctx;
    size_t bytes;
    NDArrayStorageType stype;
  };
  std::vector<PoolEntry> pool_info;

  // assign array to head gradient
  for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) {
    uint32_t nid = idx.input_nodes().at(i);
    uint32_t oid = head_grad_map_.at(idx[nid].source);
    uint32_t eid = idx.entry_id(idx.outputs()[oid]);
    NDArrayStorageType stype = (NDArrayStorageType) vstorage_type[eid];
    bool unknown_shape = !shape_is_known(vshape[eid]);
    CHECK_NE(vdtype[eid], -1);
    auto data_eid = idx.entry_id(nid, 0);
    // initialize based on storage_type
    if (stype != kDefaultStorage) {
      data_entry_[data_eid] = NDArray(stype, vshape[eid], data_context[eid], true, vdtype[eid]);
    } else if (!unknown_shape) {
      data_entry_[data_eid] = NDArray(vshape[eid], data_context[eid], false, vdtype[eid]);
    } else {
      data_entry_[data_eid] = NDArray(data_context[eid], vdtype[eid]);
    }
    if (log_verbose_) {
      LOG(INFO) << "\tinit head_grad entry\t" << data_eid << "\tas "
                << common::stype_string(stype);
    }
  }
  // get maximum bytes in each pool
  for (size_t i = 0; i < vshape.size(); ++i) {
    if (!data_entry_[i].is_none()) continue;
    size_t shape_size = 0;
    if (shape_is_known(vshape[i])) {
      shape_size = vshape[i].Size();
    }
    size_t bytes = shape_size * mshadow::mshadow_sizeof(vdtype[i]);
    int storage_id = vstorage[i];
    // skip pool allocation for kBadStorageID, kExternalStorageID and kDynamicStorageID
    if (storage_id < 0) continue;
    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_info.size()) {
      pool_info.resize(sid + 1, PoolEntry{Context::CPU(), size_t(0), kUndefinedStorage});
    }
    PoolEntry& info = pool_info[sid];
    if (info.bytes == 0) {
      info = PoolEntry{data_context[i], bytes, data_storage_type[i]};
    } else {
      info.bytes = std::max(info.bytes, bytes);
    }
  }
  // construct the re-use pool, if needed
  std::multimap<size_t, NDArray> free_pool;
  if (shared_pool != nullptr) {
    for (const NDArray& nd : *shared_pool) {
      size_t bytes = 0;
      if (shape_is_known(nd.shape())) {
        bytes = nd.shape().Size() * mshadow::mshadow_sizeof(nd.dtype());
      }
      free_pool.insert(std::make_pair(bytes, nd));
    }
  }
  // remake the data pool
  data_pool_.clear();
  data_pool_.resize(pool_info.size());

  // sort the pool info the descending order before allocating memory
  std::vector<size_t> sorted_pool_index;
  for (size_t i = 0; i < pool_info.size(); i++) {
    sorted_pool_index.push_back(i);
  }
  auto pool_comparator = [&pool_info](size_t lhs, size_t rhs){
    return pool_info[lhs].bytes > pool_info[rhs].bytes;
  };
  std::sort(sorted_pool_index.begin(), sorted_pool_index.end(), pool_comparator);

  for (size_t i : sorted_pool_index) {
    const Context& ctx = pool_info[i].ctx;
    size_t bytes = pool_info[i].bytes;
    bool allocated = false;
    for (auto it = free_pool.lower_bound(bytes); it != free_pool.end(); ++it) {
      if (it->second.ctx() == ctx && it->first >= bytes) {
        data_pool_[i] = it->second;
        free_pool.erase(it);
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      size_t nword = (bytes + 3) / 4;
      CHECK_LE(nword, std::numeric_limits<nnvm::dim_t>::max());
      // allocate float arrays
      mxnet::TShape shape{static_cast<nnvm::dim_t>(nword)};
      // TODO(junwu): adding delay_alloc=true to create nd
      // is a temporary solution.
      NDArray nd(shape, ctx, true);
      data_pool_[i] = nd;
      // put the new allocated arrays to shared pool
      if (shared_pool != nullptr)  {
        shared_pool->push_back(nd);
      }
    }
  }
  CHECK_EQ(data_pool_.size(), pool_info.size());
  // assign the data entries
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    // avoid pre-allocated arrays
    if (!data_entry_[i].is_none()) continue;
    // assign allocated array by storage id
    int storage_id = vstorage[i];
    auto storage_type = (NDArrayStorageType) vstorage_type[i];
    if (storage_type == kDefaultStorage) {
      if (!shape_is_known(vshape[i])) {
        data_entry_[i] = NDArray(data_context[i], vdtype[i]);
      } else {
        CHECK_GE(storage_id, 0) << "Do not support runtime shape op yet";
        const NDArray& src = data_pool_.at(storage_id);
        data_entry_[i] = src.AsArray(vshape[i], vdtype[i]);
      }
    } else {
      data_entry_[i] = NDArray(storage_type, vshape[i], data_context[i],
                               true, vdtype[i]);
    }
    if (log_verbose_) {
      LOG(INFO) << "\tinit data entry\t" << i << "\tas " << common::stype_string(storage_type);
    }
  }
}


void GraphExecutor::InitCachedOps() {
  // get the graph
  const auto& idx = graph_.indexed_graph();
  const auto& vstorage_inplace =
      graph_.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs =
      graph_.GetAttr<OpExecVector>("op_execs");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  const auto& addto_entry = graph_.GetAttr<std::vector<int> >("addto_entry");
  const auto& skip_plus_node = graph_.GetAttr<std::vector<int> >("skip_plus_node");

  op_nodes_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    op_nodes_[nid].opr_name = inode.source->op()->name.c_str();
    if (skip_plus_node.at(nid)) {
      op_nodes_[nid].skip_exec_node = true; continue;
    }

    op_nodes_[nid].exec = op_execs[nid];
    op_nodes_[nid].ctx = vctx[nid];
    auto& exec = op_nodes_[nid].exec;
    CHECK_EQ(exec->in_array.size(), 0U);
    CHECK_EQ(exec->out_array.size(), 0U);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    // detect inplace requirement
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(data_entry_[eid]);
      if (addto_entry.at(eid) != 0) {
        exec->req.push_back(kAddTo);
      } else if (vstorage_inplace[eid] >= 0) {
        exec->req.push_back(kWriteInplace);
      } else if (vstorage_inplace[eid] == -2) {
        // -2 indicate that the entry is never referenced.
        exec->req.push_back(kNullOp);
      } else {
        exec->req.push_back(kWriteTo);
      }
    }
  }
  // Note that this modifies the requirement of kWriteInplace
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    auto& e = idx.outputs()[j];
    op_nodes_[e.node_id].exec->req[e.index] =
        grad_store_[j - num_forward_outputs_].first;
  }
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (op_nodes_[nid].skip_exec_node) continue;
    auto& exec = op_nodes_[nid].exec;
    bool is_async = op_nodes_[nid].exec->exec_type() == ExecType::kAsync;
    bool is_gpu = op_nodes_[nid].ctx.dev_mask() == gpu::kDevMask;

    // the variables
    std::vector<Engine::VarHandle> use_vars, mutate_vars;
    for (const auto& nd : exec->in_array) {
      use_vars.push_back(nd.var());
    }
    for (const auto& r : exec->op_ctx.requested) {
      mutate_vars.push_back(r.var);
    }
    for (const auto& nd : exec->out_array) {
      mutate_vars.push_back(nd.var());
    }
    if (exec->var() != nullptr) {
      mutate_vars.push_back(exec->var());
    }
    // dedup vars
    Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
    // all vars include both mutate vars and use vars
    std::vector<Engine::VarHandle> all_vars(use_vars);
    std::copy(mutate_vars.begin(), mutate_vars.end(),
              std::inserter(all_vars, all_vars.end()));
    // setup exec vars
    Engine::Get()->PushAsync(
      [exec](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        exec->Setup();
        on_complete();
      }, Context::CPU(), {}, all_vars, FnProperty::kNormal, 0,
      "SetupExec");
    auto exec_fun = [exec, is_async, is_gpu] (
        RunContext ctx, Engine::CallbackOnComplete on_complete) {
      if (is_async) {
        exec->op_ctx.async_on_complete = on_complete;
      }
      exec->Run(ctx, is_gpu);
      // call on complete only if it is async op
      if (!is_async) {
        if (is_gpu) {
        #if MXNET_USE_CUDA
          // Wait GPU kernel to finish.
          ctx.get_stream<gpu>()->Wait();
        #else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        #endif
        }
        on_complete();
      }
    };
    // setup the vars
    op_nodes_[nid].cached_opr = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
        op_nodes_[nid].opr_name);
    op_nodes_[nid].mutate_vars = mutate_vars;
    op_nodes_[nid].use_vars = use_vars;
  }
}

void GraphExecutor::InitOpSegs() {
  size_t total_num_nodes = graph_.indexed_graph().num_nodes();
  cached_seg_opr_.clear();
  CachedSegOpr p;
  cached_seg_opr_.resize(total_num_nodes, p);
  if (monitor_callback_) return;

  // Symbolic bulking is set by the same environment variables as Imperative bulking.
  // Generate segments based on the graph structure
  bool prefer_bulk_exec_inference = Imperative::PreferBulkExecInference();
  // Whether to perform bulk exec for training
  const profiler::Profiler *prof = profiler::Profiler::Get();
  bool prefer_bulk_exec_train = Imperative::PreferBulkExecTrain()
                                && (!prof || !prof->AggregateEnabled());
  if (this->is_dynamic_) {
    prefer_bulk_exec_inference = false;
    prefer_bulk_exec_train = false;
  }
  bool is_training = num_forward_nodes_ != total_num_nodes;

  if (prefer_bulk_exec_train && is_training) {
    // Bulk the forward portion of the graph per the bulk segment max size for forward training
    this->BulkOpSegs(0, num_forward_nodes_, Imperative::BulkExecMaxNodeTrainFwd());
    // Bulk the backward portion of the graph per the bulk segment max size for backward training
    this->BulkOpSegs(num_forward_nodes_, total_num_nodes, Imperative::BulkExecMaxNodeTrainBwd());
  }

  if (prefer_bulk_exec_inference && !is_training) {
    // Bulk the entire graph as one bulk segment if possible
    this->BulkOpSegs(0, total_num_nodes, total_num_nodes);
  }
}


void GraphExecutor::BulkOpSegs(size_t from_node, size_t up_to_node, size_t segment_num_nodes_max) {
  size_t topo_start = from_node;
  size_t segment_node_count = 0;
  for (size_t nid = from_node; nid < up_to_node; nid++) {
    auto &node = graph_.indexed_graph()[nid].source;
    auto &op_node = op_nodes_[nid];
    // Variables, such as learned weights, are ignored in the segment_node_count
    bool ignore_node = node->is_variable() || op_node.skip_exec_node || op_node.exec == nullptr;
    if (!ignore_node)
      segment_node_count++;
    bool can_bulk = ignore_node || op_node.exec->exec_type() == ExecType::kSync;
    // check if we need to create the segment based on properties of this node
    if (!can_bulk || nid == up_to_node - 1 || segment_node_count >= segment_num_nodes_max) {
      // Create a new segment for the previous nodes- include also this node if it's bulkable
      cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, can_bulk ? nid + 1 : nid);
      topo_start = nid + 1;
      segment_node_count = 0;
    }
  }
}

void GraphExecutor::ExecuteMonInputCallback(size_t nid) {
  static const auto& flist_inputs =
      nnvm::Op::GetAttr<nnvm::FListInputNames>("FListInputNames");
  const auto& idx = graph_.indexed_graph();
  std::vector<std::string> input_names;
  OpNode& opnode = op_nodes_[nid];
  const auto& inode = idx[nid];
  const auto& node = idx[nid].source;
  if (flist_inputs.count(node->op())) {
    input_names = flist_inputs[node->op()](node->attrs);
  } else {
    for (size_t i = 0; i < node->num_inputs(); ++i) {
      input_names.emplace_back("input" + std::to_string(i));
    }
  }
  CHECK_EQ(opnode.exec->in_array.size(), input_names.size());
  for (size_t i = 0; i < opnode.exec->in_array.size(); ++i) {
    if (node->inputs[i].node->is_variable()) {
    // Monitor variable
    NDArray *cpy = new NDArray(opnode.exec->in_array[i]);
    std::string name = node->inputs[i].node->attrs.name;
    this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
    }
    NDArray *cpy = new NDArray(opnode.exec->in_array[i]);
    std::string name = inode.source->attrs.name + "_" + input_names[i];
    this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
  }
}

void GraphExecutor::ExecuteMonOutputCallback(size_t nid) {
  const auto& idx = graph_.indexed_graph();
  OpNode& opnode = op_nodes_[nid];
  const auto& node = idx[nid].source;
  for (size_t i = 0; i < opnode.exec->out_array.size(); ++i) {
    NDArray *cpy = new NDArray(opnode.exec->out_array[i]);
    nnvm::ObjectPtr node_ptr = std::make_shared<nnvm::Node>(*node);
    std::string name = GetOutputName({node_ptr, static_cast<uint32_t >(i), 0});
    this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  static auto& finfer_shape = nnvm::Op::GetAttr<mxnet::FInferShape>("FInferShape");
  static auto& is_backward = Op::GetAttr<nnvm::TIsBackward>("TIsBackward");
  // Update context
  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    OpNode& opnode = op_nodes_[nid];
    if (opnode.skip_exec_node) continue;
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    opnode.exec->op_ctx.is_train = is_train;
    opnode.exec->op_ctx.need_grad = need_grad_;
  }

  mxnet::ShapeVector rshape = graph_.MoveCopyAttr<mxnet::ShapeVector>("shape");
  // Push Ops
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    auto seg_op = cached_seg_opr_[nid];
    // Check segments first
    if (monitor_callback_ == nullptr && seg_op.opr != nullptr && seg_op.topo_end <= topo_end) {
      bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
      Engine::Get()->Push(seg_op.opr, seg_op.ctx, 0, profiling);
      nid = seg_op.topo_end - 1;
      continue;
    }
    // Normal mode
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    if (op_nodes_[nid].skip_exec_node) continue;
    // Monitor callbacks
    if (monitor_callback_ && monitor_all_) {
      ExecuteMonInputCallback(nid);
    }
    if (this->is_dynamic_) {
      const auto &op = inode.source->op();
      {
        for (NDArray &array : opnode.exec->in_array) {
          array.WaitToRead();
          if (!shape_is_known(array.shape())) {
            array.SetShapeFromChunk();
          }
        }
        int i = 0;
        for (NDArray &array : opnode.exec->out_array) {
          array.WaitToRead();
          if (!shape_is_known(array.shape())) {
            array.SetShapeFromChunk();
          }
          if (!shape_is_known(array.shape())) {
            mxnet::TShape shape = rshape[idx.entry_id(nid, i)];
            if (shape_is_known(shape)) {
              array.ReshapeAndAlloc(shape);
            }
          }
          ++i;
        }
      }
      if (finfer_shape.count(op)) {
        mxnet::ShapeVector in_shapes;
        mxnet::ShapeVector out_shapes;
        for (NDArray &array : opnode.exec->in_array) {
          in_shapes.push_back(array.shape());
        }
        for (NDArray &array : opnode.exec->out_array) {
          out_shapes.push_back(array.shape());
        }
        auto finfer = finfer_shape[op];
        try {
          bool success = finfer(inode.source->attrs, &in_shapes, &out_shapes);
          CHECK(success) << "InferShape failed in operator " << inode.source->attrs.name;
        } catch (const std::exception& e) {
          throw dmlc::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
        }
        int n_out = out_shapes.size();
        for (int i = 0; i < n_out; ++i) {
          NDArray &array = opnode.exec->out_array[i];
          if (!shape_is_known(array.shape())) {
            array.Init(out_shapes[i]);
          }
        }
      } else if (is_backward.get(inode.source->op(), false) && inode.control_deps.size()) {
        CHECK_GE(inode.control_deps.size(), 1U) <<
          "BackwardOp need to have control_deps to its forward op";
        uint32_t fid = inode.control_deps[0];
        const OpNode& fopnode = op_nodes_[fid];
        CHECK_EQ(fopnode.exec->in_array.size(), opnode.exec->out_array.size());
        int nelem = fopnode.exec->in_array.size();
        std::vector<NDArray> &from = fopnode.exec->in_array;
        std::vector<NDArray> &to = opnode.exec->out_array;
        for (int i = 0; i < nelem; ++i) {
          if (!shape_is_known(to[i].shape())) {
            to[i].Init(from[i].shape());
          }
        }
      }
    }
    opnode.exec->op_ctx.is_train = is_train;
    opnode.exec->op_ctx.need_grad = need_grad_;
    if (opnode.exec->exec_type() == ExecType::kCrossDeviceCopy) {
      CHECK_EQ(inode.inputs.size(), 1U);
      CHECK_EQ(opnode.exec->in_array.size(), 1U);
      CHECK_EQ(opnode.exec->out_array.size(), 1U);
      CopyFromTo(opnode.exec->in_array[0], &(opnode.exec->out_array[0]));
    } else if (opnode.exec->exec_type() == ExecType::kSubgraphExec) {
      // If the node contains a subgraph, we can't execute it in the engine.
      opnode.exec->Run(opnode.exec->op_ctx.run_ctx, false);
    } else if (opnode.cached_opr != nullptr) {
      bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx, 0, profiling);
      if (this->is_dynamic_) {
        for (NDArray &array : opnode.exec->out_array) {
          array.WaitToRead();
          if (!shape_is_known(array.shape())) {
            array.SetShapeFromChunk();
          }
        }
      }
    } else {
      LOG(FATAL) << "Not accessed";
    }
    for (uint32_t i = 0; i < num_inputs; ++i) {
      int eid = idx.entry_id(inode.inputs[i]);
      if (!shape_is_known(rshape[eid])) {
        rshape[eid] = opnode.exec->in_array[i].shape();
      }
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
      int eid = idx.entry_id(nid, i);
      if (!shape_is_known(rshape[eid])) {
        rshape[eid] = opnode.exec->out_array[i].shape();
      }
    }
    // Monitor callbacks
    if (monitor_callback_) {
      ExecuteMonOutputCallback(nid);
    }
  }
  graph_.attrs["shape"] = std::make_shared<dmlc::any>(rshape);
}

GraphExecutor::CachedSegOpr GraphExecutor::CreateCachedSegOpr(size_t topo_start, size_t topo_end) {
  std::vector<Engine::VarHandle> use_vars;
  std::vector<Engine::VarHandle> mutate_vars;
  Context *pctx = nullptr;
  GraphExecutor::CachedSegOpr ret;
  ret.topo_start = topo_start;
  ret.topo_end = topo_end;
  auto& exec_list = ret.exec_list;
  // invalid segment
  if (topo_end <= topo_start) {
    return ret;
  }
  std::string opr_names = "[";

  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    std::vector<Engine::VarHandle> all_vars;
    const auto& inode = idx[nid];
    OpNode& op_node = op_nodes_[nid];
    if (op_node.skip_exec_node) continue;
    if (inode.source->is_variable()) continue;
    if (op_node.exec->exec_type() != ExecType::kSync) {
      return ret;
    }
    if (pctx == nullptr) pctx = &(op_node.ctx);
    if (*pctx != op_node.ctx) {
      return ret;
    }
    auto& exec = op_nodes_[nid].exec;
    std::copy(op_node.mutate_vars.begin(), op_node.mutate_vars.end(),
              std::inserter(mutate_vars, mutate_vars.end()));
    std::copy(op_node.use_vars.begin(), op_node.use_vars.end(),
              std::inserter(use_vars, use_vars.end()));
    ret.exec_list.push_back(exec);
    opr_names += inode.source->op()->name + ",";
  }

  if (pctx == nullptr) return ret;
  ret.ctx = *pctx;
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);

  bool is_gpu = pctx->dev_mask() == gpu::kDevMask;
  auto exec_fun = [exec_list, is_gpu] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    // Run all opr in the sub-graph
    for (auto &exec : exec_list) {
      exec->Run(ctx, is_gpu);
    }
    if (is_gpu) {
#if MXNET_USE_CUDA
      // Wait GPU kernel to finish.
      ctx.get_stream<gpu>()->Wait();
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    on_complete();
  };
  opr_names.pop_back();
  opr_names += "]";
  ret.opr = Engine::Get()->NewOperator(
    exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
    opr_names.c_str());
  return ret;
}

// Infer shapes, dtypes, stypes, contexts for the forward graph
static nnvm::Graph InferForwardAttrs(nnvm::Graph g,
                                     mxnet::ShapeVector arg_shapes,
                                     nnvm::DTypeVector arg_dtypes,
                                     StorageTypeVector arg_stypes,
                                     const Context& default_ctx,
                                     const std::map<std::string, Context>& ctx_map,
                                     const std::vector<Context>& in_arg_ctxes,
                                     const std::vector<Context>& aux_state_ctxes,
                                     bool partial_shape = false) {
  const auto& indexed_graph = g.indexed_graph();
  const auto num_forward_inputs = indexed_graph.input_nodes().size();
  g = AssignContext(g, default_ctx, ctx_map, in_arg_ctxes, {},
                   aux_state_ctxes, {}, num_forward_inputs, g.outputs.size());
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    if (!partial_shape) {
      HandleInferShapeError(num_forward_inputs, indexed_graph,
                            g.GetAttr<mxnet::ShapeVector>("shape"));
    }
  }
  g = InferType(std::move(g), std::move(arg_dtypes), "__dtype__");
  if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) {
    HandleInferTypeError(num_forward_inputs, indexed_graph,
                         g.GetAttr<nnvm::DTypeVector>("dtype"));
  }
  g = InferStorageType(std::move(g), std::move(arg_stypes), "__storage_type__");
  if (g.GetAttr<size_t>("storage_type_num_unknown_nodes") != 0U) {
    HandleInferStorageTypeError(num_forward_inputs, indexed_graph,
                                g.GetAttr<StorageTypeVector>("storage_type"));
  }
  return g;
}

static bool SubgraphBackendCheck(const op::SubgraphBackendPtr& backend,
                                 const Context& default_ctx,
                                 int verbose = 1) {
  if (backend->HasAttr("enable") && (backend->GetAttr<bool>("enable") != true)) {
    if (verbose > 1) {
      LOG(INFO) << "Subgraph backend " << backend->GetName()
                << " isn't activated.";
    }
    return false;
  }
  if (backend->HasAttr("context") && backend->GetAttr<Context>("context") != default_ctx) {
    if (verbose > 1) {
      LOG(INFO) << "Subgraph backend " << backend->GetName()
                << " isn't activated as context mismatch.";
    }
    return false;
  }
  return true;
}

static bool SubgraphPropertyCheck(const std::string& backend_name,
                                  const op::SubgraphPropertyPtr& prop, bool need_grad,
                                  int verbose = 1) {
  auto full_name =
      prop->HasAttr("property_name") ? prop->GetAttr<std::string>("property_name") : std::string();
  if (prop->HasAttr("disable") && prop->GetAttr<bool>("disable") == true) {
    LOG(INFO) << "subgraph property " << full_name << " from backend " << backend_name
              << " is disabled.";
    return false;
  }
  if (prop->HasAttr("inference_only") && prop->GetAttr<bool>("inference_only") == true) {
    if (need_grad) {
      if (verbose > 1) {
        LOG(INFO) << "skip partitioning graph with subgraph property " << full_name
                  << " from backend " << backend_name << " as it requires `grad_req=null`.";
      }
      return false;
    }
  }
  return true;
}

// Given input attr arrays, partition the graph using the backend name equal to prop_name.
// This is a common function for bind and simple_bind flows.
static nnvm::Symbol BuildSubgraph(const nnvm::Symbol& src, op::SubgraphPropertyPtr subgraph_prop,
                                  const mxnet::ShapeVector& arg_shapes,
                                  const nnvm::DTypeVector& arg_dtypes,
                                  const StorageTypeVector& arg_stypes, const Context& default_ctx,
                                  const std::map<std::string, Context>& ctx_map,
                                  const std::vector<Context>& in_arg_ctxes,
                                  const std::vector<Context>& aux_state_ctxes) {
  nnvm::Symbol ret = src.Copy();
  nnvm::Graph g;
  g.outputs = ret.outputs;
  g = InferForwardAttrs(g, arg_shapes, arg_dtypes, arg_stypes, default_ctx, ctx_map, in_arg_ctxes,
                        aux_state_ctxes, true);
  subgraph_prop->SetAttr("graph", g);
  g.attrs["subgraph_property"] = std::make_shared<nnvm::any>(subgraph_prop);
  g = ApplyPass(std::move(g), "BuildSubgraph");
  subgraph_prop->RemoveAttr("graph");
  g.attrs.erase("subgraph_property");
  ret.outputs = g.outputs;
  return ret;
}

// Given input attr dicts, partition the graph using the backend.
// This is for simple_bind flow.
static nnvm::Symbol BuildSubgraph(
    const nnvm::Symbol& src, const op::SubgraphBackendPtr backend,
    const std::unordered_map<std::string, mxnet::TShape>& arg_shape_map,
    const std::unordered_map<std::string, int>& arg_dtype_map,
    const std::unordered_map<std::string, int>& arg_stype_map, const Context& default_ctx,
    const std::map<std::string, Context>& ctx_map, std::vector<Context>* in_arg_ctxes,
    std::vector<Context>* arg_grad_ctxes, std::vector<OpReqType>* grad_req_types,
    std::vector<Context>* aux_state_ctxes, int verbose = 1) {
  // setup map for in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes and grad_req_types
  std::unordered_map<std::string, Context> in_arg_ctx_map;
  std::unordered_map<std::string, Context> arg_grad_ctx_map;
  std::unordered_map<std::string, Context> aux_state_ctx_map;
  std::unordered_map<std::string, OpReqType> grad_req_type_map;

  auto arg_names = src.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
  auto aux_names = src.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
  for (size_t i = 0; i < arg_names.size(); ++i) {
    const auto& name = arg_names[i];
    in_arg_ctx_map[name] = in_arg_ctxes->at(i);
    arg_grad_ctx_map[name] = arg_grad_ctxes->at(i);
    grad_req_type_map[name] = grad_req_types->at(i);
  }

  for (size_t i = 0; i < aux_names.size(); ++i) {
    aux_state_ctx_map[aux_names[i]] = aux_state_ctxes->at(i);
  }

  bool need_grad = false;
  for (OpReqType req : *grad_req_types) {
    if (req != kNullOp) {
      need_grad = true;
      break;
    }
  }
  nnvm::Symbol ret = src.Copy();
  std::unordered_set<std::string> op_names_set;
  const auto& backend_name = backend->GetName();
  const auto it = op::SubgraphPropertyOpNameSet::Get()->find(backend_name);
  // assign a op name set to the subgraph property if it has been provided by users
  if (it != op::SubgraphPropertyOpNameSet::Get()->end()) {
    LOG(INFO) << "SubgraphPropertyOpNameSet for subgraph property " << backend_name
              << " has been assigned a value. Please make sure it is initialized"
                 " only for the testing purpose.";
    op_names_set = it->second;
  }

  const auto& subgraph_prop_list = backend->GetSubgraphProperties();
  for (auto& subgraph_prop : subgraph_prop_list) {
    if (SubgraphPropertyCheck(backend_name, subgraph_prop, need_grad, verbose)) {
      subgraph_prop->SetAttr("op_names", op_names_set);
      const std::vector<std::string> input_names = ret.ListInputNames(Symbol::kAll);
      mxnet::ShapeVector arg_shapes(input_names.size(), mxnet::TShape());
      nnvm::DTypeVector arg_dtypes(input_names.size(), -1);
      StorageTypeVector arg_stypes(input_names.size(), kUndefinedStorage);
      for (size_t i = 0; i < input_names.size(); ++i) {
        const auto& input_name = input_names[i];
        const auto it1 = arg_shape_map.find(input_name);
        if (arg_shape_map.end() != it1) {
          arg_shapes[i] = it1->second;
        }
        const auto it2 = arg_dtype_map.find(input_name);
        if (arg_dtype_map.end() != it2) {
          arg_dtypes[i] = it2->second;
        }
        const auto it3 = arg_stype_map.find(input_name);
        if (arg_stype_map.end() != it3) {
          arg_stypes[i] = it3->second;
        }
      }
      ret = BuildSubgraph(ret, subgraph_prop, arg_shapes, arg_dtypes, arg_stypes, default_ctx,
                          ctx_map, *in_arg_ctxes, *aux_state_ctxes);
      // Reorder in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes and grad_req_types according to
      // partitioned symbol input sequence
      in_arg_ctxes->clear();
      arg_grad_ctxes->clear();
      aux_state_ctxes->clear();
      grad_req_types->clear();
      auto new_arg_names = ret.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
      auto new_aux_names = ret.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      for (const auto& arg_name : new_arg_names) {
        CHECK(in_arg_ctx_map.count(arg_name));
        in_arg_ctxes->push_back(in_arg_ctx_map[arg_name]);
        arg_grad_ctxes->push_back(arg_grad_ctx_map[arg_name]);
        grad_req_types->push_back(grad_req_type_map[arg_name]);
      }
      for (const auto& arg_name : new_aux_names) {
        CHECK(aux_state_ctx_map.count(arg_name));
        aux_state_ctxes->push_back(aux_state_ctx_map[arg_name]);
      }
    }
  }
  return ret;
}

// Given input ndarrays, partition the graph using backend.
// This is for bind flow.
static nnvm::Symbol BuildSubgraph(const nnvm::Symbol& src, const op::SubgraphBackendPtr backend,
                                  const Context& default_ctx,
                                  const std::map<std::string, Context>& ctx_map,
                                  std::vector<NDArray>* in_args,
                                  std::vector<NDArray>* arg_grad_store,
                                  std::vector<OpReqType>* grad_req_type,
                                  std::vector<NDArray>* aux_states, int verbose = 1) {
  // setup map for in_args, arg_grad_store, grad_req_type and aux_states
  std::unordered_map<std::string, NDArray> in_args_map;
  std::unordered_map<std::string, NDArray> arg_grad_store_map;
  std::unordered_map<std::string, OpReqType> grad_req_type_map;
  std::unordered_map<std::string, NDArray> aux_states_map;
  const std::vector<std::string> arg_names = src.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
  const std::vector<std::string> aux_names = src.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
  for (size_t i = 0; i < arg_names.size(); ++i) {
    in_args_map[arg_names[i]] = in_args->at(i);
  }

  for (size_t i = 0; i < aux_names.size(); ++i) {
    aux_states_map[aux_names[i]] = aux_states->at(i);
  }

  if (arg_grad_store->size()) {
    for (size_t i = 0; i < arg_names.size(); ++i) {
      const auto& name = arg_names[i];
      arg_grad_store_map[name] = arg_grad_store->at(i);
      grad_req_type_map[name] = grad_req_type->at(i);
    }
  }

  bool need_grad = false;
  for (OpReqType req : *grad_req_type) {
    if (req != kNullOp) {
      need_grad = true;
      break;
    }
  }
  nnvm::Symbol ret = src.Copy();
  std::unordered_set<std::string> op_names_set;
  const auto& backend_name = backend->GetName();
  auto it = op::SubgraphPropertyOpNameSet::Get()->find(backend_name);
  // assign a op name set to the subgraph property if it has been provided by users
  if (it != op::SubgraphPropertyOpNameSet::Get()->end()) {
    LOG(INFO) << "SubgraphPropertyOpNameSet for subgraph property " << backend_name
              << " has been assigned a value. Please make sure it is initialized"
                 " only for the testing purpose.";
    op_names_set = it->second;
  }
  const auto& subgraph_prop_list = backend->GetSubgraphProperties();

  for (auto subgraph_prop : subgraph_prop_list) {
    if (SubgraphPropertyCheck(backend_name, subgraph_prop, need_grad, verbose)) {
      subgraph_prop->SetAttr("op_names", op_names_set);
      const std::vector<std::string> input_names = ret.ListInputNames(Symbol::kAll);
      const std::vector<std::string> arg_names = ret.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
      const std::vector<std::string> aux_names = ret.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      CHECK_EQ(arg_names.size(), in_args_map.size());
      CHECK_EQ(aux_names.size(), aux_states_map.size());
      mxnet::ShapeVector arg_shapes;  // all input shapes
      arg_shapes.reserve(input_names.size());
      nnvm::DTypeVector arg_dtypes;  // all input dtypes
      arg_dtypes.reserve(input_names.size());
      StorageTypeVector arg_stypes;  // all input stypes
      arg_stypes.reserve(input_names.size());
      std::vector<Context> in_arg_ctxes(in_args_map.size());
      std::vector<Context> aux_state_ctxes(aux_states_map.size());

      size_t i1 = 0, i2 = 0;
      for (const auto& input_name : input_names) {
        if (i2 < aux_names.size() && aux_names[i2] == input_name) {
          const auto &aux_st = aux_states_map[input_name];
          arg_shapes.push_back(aux_st.shape());
          arg_dtypes.push_back(aux_st.dtype());
          arg_stypes.push_back(aux_st.storage_type());
          aux_state_ctxes[i2] = aux_st.ctx();
          ++i2;
        } else {
          CHECK(i1 < arg_names.size());
          CHECK_EQ(arg_names[i1], input_name);
          const auto &in_arg = in_args_map[input_name];
          arg_shapes.push_back(in_arg.shape());
          arg_dtypes.push_back(in_arg.dtype());
          arg_stypes.push_back(in_arg.storage_type());
          in_arg_ctxes[i1] = in_arg.ctx();
          ++i1;
        }
      }

      ret = BuildSubgraph(ret, subgraph_prop, arg_shapes, arg_dtypes, arg_stypes, default_ctx,
                          ctx_map, in_arg_ctxes, aux_state_ctxes);
    }
  }
  // Reorder in_args, arg_grad_store, grad_req_type and aux_states according to partitioned symbol
  // input sequence
  const auto new_arg_names = ret.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
  const auto new_aux_names = ret.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
  CHECK_EQ(arg_names.size(), new_arg_names.size());
  CHECK_EQ(arg_names.size(), new_arg_names.size());
  in_args->clear();
  aux_states->clear();
  for (const auto& arg_name : new_arg_names) {
    CHECK(in_args_map.count(arg_name));
    in_args->push_back(in_args_map[arg_name]);
  }

  for (const auto& arg_name : new_aux_names) {
    CHECK(aux_states_map.count(arg_name));
    aux_states->push_back(aux_states_map[arg_name]);
  }

  if (arg_grad_store->size()) {
    arg_grad_store->clear();
    grad_req_type->clear();
    for (const auto& arg_name : new_arg_names) {
      arg_grad_store->push_back(arg_grad_store_map[arg_name]);
      grad_req_type->push_back(grad_req_type_map[arg_name]);
    }
  }
  return ret;
}
}  // namespace exec

Executor *Executor::SimpleBind(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& group2ctx,
                               const std::vector<Context>& in_arg_ctxes,
                               const std::vector<Context>& arg_grad_ctxes,
                               const std::vector<Context>& aux_state_ctxes,
                               const std::unordered_map<std::string, mxnet::TShape>& arg_shape_map,
                               const std::unordered_map<std::string, int>& arg_dtype_map,
                               const std::unordered_map<std::string, int>& arg_stype_map,
                               const std::vector<OpReqType>& grad_req_types,
                               const std::unordered_set<std::string>& shared_arg_names,
                               std::vector<NDArray>* in_args,
                               std::vector<NDArray>* arg_grads,
                               std::vector<NDArray>* aux_states,
                               std::unordered_map<std::string, NDArray>* shared_buffer,
                               Executor* shared_exec) {
  auto exec = new exec::GraphExecutor(symbol);
  bool init = false;
  if (!exec->subgraph_property().empty()) {
    static int verbose = dmlc::GetEnv("MXNET_SUBGRAPH_VERBOSE", 1);
    const auto& backend_name = exec->subgraph_property();
    const auto& backend = op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(backend_name);
    if (exec::SubgraphBackendCheck(backend, default_ctx, verbose)) {
      if (verbose) LOG(INFO) << "Subgraph backend " << backend_name << " is activated.";
      std::vector<Context> tmp_in_arg_ctxes = in_arg_ctxes;
      std::vector<Context> tmp_arg_grad_ctxes = arg_grad_ctxes;
      std::vector<Context> tmp_aux_state_ctxes = aux_state_ctxes;
      std::vector<OpReqType> tmp_grad_req_types = grad_req_types;
      std::vector<NDArray> tmp_in_args;
      std::vector<NDArray> tmp_arg_grads;
      std::vector<NDArray> tmp_aux_states;
      const auto arg_names = symbol.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
      const auto aux_names = symbol.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      symbol = exec::BuildSubgraph(symbol, backend, arg_shape_map, arg_dtype_map, arg_stype_map,
                                   default_ctx, group2ctx, &tmp_in_arg_ctxes, &tmp_arg_grad_ctxes,
                                   &tmp_grad_req_types, &tmp_aux_state_ctxes, verbose);
      // Subgraph cannot be recreated from unoptimized symbol
      exec = new exec::GraphExecutor(symbol);
      exec->Init(symbol.Copy(), default_ctx, group2ctx, tmp_in_arg_ctxes, tmp_arg_grad_ctxes,
                 tmp_aux_state_ctxes, arg_shape_map, arg_dtype_map, arg_stype_map,
                 tmp_grad_req_types, shared_arg_names, &tmp_in_args, &tmp_arg_grads,
                 &tmp_aux_states, shared_buffer, shared_exec);
      init = true;
      const auto new_arg_names = symbol.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
      const auto new_aux_names = symbol.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      std::unordered_map<std::string, size_t> new_arg_names_idx_map;
      std::unordered_map<std::string, size_t> new_aux_names_idx_map;
      for (size_t i = 0; i != new_arg_names.size(); ++i) {
        new_arg_names_idx_map[new_arg_names[i]] = i;
      }
      for (size_t i = 0; i != new_aux_names.size(); ++i) {
        new_aux_names_idx_map[new_aux_names[i]] = i;
      }

      in_args->reserve(arg_names.size());
      arg_grads->reserve(arg_names.size());
      for (size_t i = 0; i != arg_names.size(); ++i) {
        const auto& arg_name = arg_names[i];
        const auto& it = new_arg_names_idx_map.find(arg_name);
        CHECK(it != new_arg_names_idx_map.end())
            << "Subgraph doesn't support remove any input node for now.";
        in_args->emplace_back(std::move(tmp_in_args[it->second]));
        arg_grads->emplace_back(std::move(tmp_arg_grads[it->second]));
      }

      aux_states->reserve(aux_names.size());
      for (size_t i = 0; i != aux_names.size(); ++i) {
        const auto& aux_name = aux_names[i];
        const auto& it = new_aux_names_idx_map.find(aux_name);
        CHECK(it != new_aux_names_idx_map.end())
            << "Subgraph doesn't support remove any input node for now.";
        aux_states->emplace_back(std::move(tmp_aux_states[it->second]));
      }
    }
  }
  if (!init) {
    // init without subgraph
    exec->Init(symbol.Copy(), default_ctx, group2ctx, in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,
               arg_shape_map, arg_dtype_map, arg_stype_map, grad_req_types, shared_arg_names,
               in_args, arg_grads, aux_states, shared_buffer, shared_exec);
  }
  return exec;
}

Executor *Executor::Bind(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& group2ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states,
                         Executor* shared_exec) {
  auto exec = new exec::GraphExecutor(symbol);
  static int verbose = dmlc::GetEnv("MXNET_SUBGRAPH_VERBOSE", 1);
  std::vector<NDArray> tmp_in_args = in_args;
  std::vector<NDArray> tmp_arg_grad_store = arg_grad_store;
  std::vector<OpReqType> tmp_grad_req_type = grad_req_type;
  std::vector<NDArray> tmp_aux_states = aux_states;

  if (!exec->subgraph_property().empty()) {
    const auto& backend_name = exec->subgraph_property();
    const auto& backend = op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(backend_name);
    if (exec::SubgraphBackendCheck(backend, default_ctx, verbose)) {
      if (verbose) LOG(INFO) << "Subgraph backend " << backend_name << " is activated.";
      symbol = exec::BuildSubgraph(symbol, backend, default_ctx, group2ctx, &tmp_in_args,
                                   &tmp_arg_grad_store, &tmp_grad_req_type, &tmp_aux_states,
                                   verbose);
      // Subgraph cannot be recreated from unoptimized symbol
      exec = new exec::GraphExecutor(symbol);
    }
  }
  exec->Init(symbol.Copy(), default_ctx, group2ctx, tmp_in_args, tmp_arg_grad_store,
             tmp_grad_req_type, tmp_aux_states, reinterpret_cast<Executor*>(shared_exec));
  return exec;
}
}  // namespace mxnet
