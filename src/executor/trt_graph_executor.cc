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

#if MXNET_USE_TENSORRT

#include "trt_graph_executor.h"

#include <onnx/onnx.pb.h>
#include <NvInfer.h>
#include "./onnx_to_tensorrt.h"
#include "../operator/contrib/tensorrt-inl.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"


namespace mxnet {
namespace exec {

using namespace mxnet::common;

  /*!
 * \brief TrtGraphExecutor initializer for simple bind flow in
 * which only certain input shapes and dtypes are provided by users.
 * The initializer uses these shapes and dtypes to perform
 * shape and dtype inferences, and then create NDArrays
 * to populate data entries of the graph. The created NDArrays
 * for in_args, arg_grads and aux_states are passed to the
 * front end to attach the created executor.
 * In front end, if the simple_bind flow is trigger by
 * _bind_ith_exec, the shared data arrays of DataParallelExecutorGroup
 * and shared executor will be taken into account in creating
 * NDArrays for in_args, arg_grads, and aux_states for reusing
 * already allocated memory.
 *
 * This version of an executor exports the computation graph to TensorRT make use of fused
 * kernels and other runtime enhancements.  TRT will compile the sub-graphs to executable fused
 * operators without intervention from the user.  Operators in the original graph that are not
 * supported by TRT will continue to be executed normally by MXNet.
 *
 */
void TrtGraphExecutor::Init(nnvm::Symbol symbol,
                            const Context& default_ctx,
                            const std::map<std::string, Context>& ctx_map,
                            std::vector<Context> *in_arg_ctxes,
                            std::vector<Context> *arg_grad_ctxes,
                            std::vector<Context> *aux_state_ctxes,
                            std::unordered_map<std::string, TShape> *arg_shape_map,
                            std::unordered_map<std::string, int> *arg_dtype_map,
                            std::unordered_map<std::string, int> *arg_stype_map,
                            std::vector<OpReqType> *grad_req_types,
                            const std::unordered_set<std::string>& shared_arg_names,
                            std::vector<NDArray>* in_arg_vec,
                            std::vector<NDArray>* arg_grad_vec,
                            std::vector<NDArray>* aux_state_vec,
                            std::unordered_map<std::string, NDArray>* shared_buffer,
                            Executor* shared_exec,
                            const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  symbol = symbol.Copy();
  nnvm::Graph g = InitGraph(symbol, default_ctx, ctx_map, *in_arg_ctxes, *arg_grad_ctxes,
                            *aux_state_ctxes, *grad_req_types);

  if (need_grad_) {
    LOG(FATAL) << "You may be attempting to use TensorRT for training.  TensorRT is an inference "
                  "only library.  To re-enable legacy MXNet graph execution, which will support "
                  "training, set the MXNET_USE_TENSORRT environment variable to 0, or call "
                  "mx.contrib.tensorrt.set_use_tensorrt(False)";
  }

  if (shared_buffer == nullptr || shared_buffer->empty()) {
    LOG(FATAL) << "MXNET_USE_TENSORRT = 1 but shared_buffer is empty. "
               << "Please provide weights and other parameters, such as "
               << "BatchNorm moments, via the shared_buffer, during simple bind call.";
  }

  // The following code of shape and dtype inferences and argument
  // initialization is for simple_bind only. Regular bind operation
  // should do this differently.

  // Initialize arg_shapes and arg_dtypes for shape and type inferences.
  // It contains all in_args and aux_states' shapes and types in a certain order.
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  nnvm::ShapeVector arg_shapes(idx.input_nodes().size(), TShape());
  nnvm::DTypeVector arg_dtypes(idx.input_nodes().size(), -1);
  StorageTypeVector arg_stypes(idx.input_nodes().size(), kUndefinedStorage);
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    auto it1 = arg_shape_map->find(name);
    if (arg_shape_map->end() != it1) {
      arg_shapes[i] = it1->second;
    }
    auto it2 = arg_dtype_map->find(name);
    if (arg_dtype_map->end() != it2) {
      arg_dtypes[i] = it2->second;
    }
    auto it3 = arg_stype_map->find(name);
    if (arg_stype_map->end() != it3) {
      arg_stypes[i] = it3->second;
    }
  }
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    HandleInferShapeError(num_forward_inputs_, g.indexed_graph(),
                          g.GetAttr<nnvm::ShapeVector>("shape"));
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

  auto trt_groups = GetTrtCompatibleSubsets(g, shared_buffer);
  for (auto trt_group : trt_groups) {
    if (trt_group.size() > 1) {
      g = ReplaceSubgraph(std::move(g), trt_group, shared_buffer);
      g = ReinitGraph(std::move(g), default_ctx, ctx_map, in_arg_ctxes, arg_grad_ctxes,
                      aux_state_ctxes, grad_req_types, arg_shape_map, arg_dtype_map,
                      arg_stype_map, shared_buffer);
    }
  }


  InitArguments(g.indexed_graph(), g.GetAttr<nnvm::ShapeVector>("shape"),
                g.GetAttr<nnvm::DTypeVector>("dtype"),
                g.GetAttr<StorageTypeVector>("storage_type"),
                *in_arg_ctxes, *arg_grad_ctxes, *aux_state_ctxes,
                *grad_req_types, shared_arg_names, shared_exec,
                shared_buffer, in_arg_vec, arg_grad_vec, aux_state_vec);

  // The above code of shape and dtype inferences and argument
  // initialization is for simple_bind only. Regular bind operation
  // should do this differently.

  // Initialize the rest attributes of the graph.
  // This function can be called by regular bind
  // operation flow as well.
  FinishInitGraph(symbol, g, shared_exec, feed_dict);
}
/*!
 * \brief Initialize in_args, arg_grads, and aux_states
 * and their data_entry_ of the executor using
 * shared_buffer from DataParallelExecutorGroup
 * and shared_exec if available.
 */
void TrtGraphExecutor::InitArguments(const nnvm::IndexedGraph& idx,
                                  const nnvm::ShapeVector& inferred_shapes,
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
    const TShape& inferred_shape = inferred_shapes[eid];
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
        auto it = shared_buffer->find(arg_name);
        if (it != shared_buffer->end()) {
          aux_state_vec->push_back(std::move(it->second.Copy(aux_state_ctxes[aux_top])));
        } else {
          aux_state_vec->push_back(std::move(InitZeros(inferred_stype, inferred_shape,
                                                       aux_state_ctxes[aux_top], inferred_dtype)));
        }
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
        auto it = shared_buffer->find(arg_name);
        if (it != shared_buffer->end()) {
          in_arg_vec->push_back(std::move(it->second.Copy(in_arg_ctxes[arg_top])));
        } else {
          in_arg_vec->push_back(std::move(InitZeros(inferred_stype, inferred_shape,
                                                    in_arg_ctxes[arg_top], inferred_dtype)));
        }
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
 * \brief This function is triggered after each tensorrt subgraph replacement pass.
 * Reset arguments of GraphExecutor::Init(...) as some variables (weights and biases)
 * are absorbed into the TRT engine it also it reruns attributes inferences accordingly
 * to the new topology.
 */
Graph TrtGraphExecutor::ReinitGraph(Graph&& g, const Context &default_ctx,
                                 const std::map<std::string, Context> &ctx_map,
                                 std::vector<Context> *in_arg_ctxes,
                                 std::vector<Context> *arg_grad_ctxes,
                                 std::vector<Context> *aux_state_ctxes,
                                 std::vector<OpReqType> *grad_req_types,
                                 std::unordered_map<std::string, TShape> *arg_shape_map,
                                 std::unordered_map<std::string, int> *arg_dtype_map,
                                 std::unordered_map<std::string, int> *arg_stype_map,
                                 std::unordered_map<std::string, NDArray> *params_map) {
  std::unordered_set<std::string> to_remove_params;
  for (auto& el : *params_map) {
    to_remove_params.insert(el.first);
  }

  DFSVisit(g.outputs, [&to_remove_params](const nnvm::NodePtr n) {
    to_remove_params.erase(n->attrs.name);
  });

  for (auto& el : to_remove_params) {
    params_map->erase(el);
    arg_shape_map->erase(el);
    arg_dtype_map->erase(el);
    arg_stype_map->erase(el);
  }
  const auto &idx = g.indexed_graph();
  num_forward_inputs_ = idx.input_nodes().size();
  in_arg_ctxes->resize(num_forward_inputs_ - idx.mutable_input_nodes().size());
  arg_grad_ctxes->resize(num_forward_inputs_ - idx.mutable_input_nodes().size());
  grad_req_types->resize(num_forward_inputs_ - idx.mutable_input_nodes().size());
  aux_state_ctxes->resize(idx.mutable_input_nodes().size());

  // create "device" and "context" attrs for the graph
  g = AssignContext(g, default_ctx, ctx_map, *in_arg_ctxes, *arg_grad_ctxes,
                    *aux_state_ctxes, *grad_req_types, num_forward_inputs_,
                    num_forward_outputs_);

  // get number of nodes used in forward pass
  num_forward_nodes_ = 0;
  for (size_t i = 0; i < num_forward_outputs_; ++i) {
    num_forward_nodes_ = std::max(
        num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  nnvm::ShapeVector arg_shapes(idx.input_nodes().size(), TShape());
  nnvm::DTypeVector arg_dtypes(idx.input_nodes().size(), -1);
  StorageTypeVector arg_stypes(idx.input_nodes().size(), kUndefinedStorage);
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string &name = idx[nid].source->attrs.name;
    auto it1 = arg_shape_map->find(name);
    if (arg_shape_map->end() != it1) {
      arg_shapes[i] = it1->second;
    }
    auto it2 = arg_dtype_map->find(name);
    if (arg_dtype_map->end() != it2) {
      arg_dtypes[i] = it2->second;
    }
    auto it3 = arg_stype_map->find(name);
    if (arg_stype_map->end() != it3) {
      arg_stypes[i] = it3->second;
    }
  }
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    HandleInferShapeError(num_forward_inputs_, g.indexed_graph(),
                          g.GetAttr<nnvm::ShapeVector>("shape"));
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

  return g;
}


/*!
 * \brief Return the "optimized" symbol contained in the graph.
 * For optimization pass such as TensorRT pass
 */
nnvm::Symbol TrtGraphExecutor::GetOptimizedSymbol() {
  Symbol ret;
  ret.outputs = std::vector<nnvm::NodeEntry>(graph_.outputs.begin(),
                                             graph_.outputs.begin() + num_forward_outputs_);
  ret = ret.Copy();
  static const Op* trt_op = Op::Get("_trt_op");
  DFSVisit(ret.outputs, [](const nnvm::NodePtr n) {
    if (n->op() == trt_op) {
      n->attrs.dict.clear();
    }
  });
  return ret;
}

Executor *TrtGraphExecutor::TensorRTBind(nnvm::Symbol symbol,
                                         const Context &default_ctx,
                                         const std::map<std::string, Context> &group2ctx,
                                         std::vector<Context> *in_arg_ctxes,
                                         std::vector<Context> *arg_grad_ctxes,
                                         std::vector<Context> *aux_state_ctxes,
                                         std::unordered_map<std::string, TShape> *arg_shape_map,
                                         std::unordered_map<std::string, int> *arg_dtype_map,
                                         std::unordered_map<std::string, int> *arg_stype_map,
                                         std::vector<OpReqType> *grad_req_types,
                                         const std::unordered_set<std::string> &param_names,
                                         std::vector<NDArray> *in_args,
                                         std::vector<NDArray> *arg_grads,
                                         std::vector<NDArray> *aux_states,
                                         std::unordered_map<std::string, NDArray> *shared_buffer,
                                         Executor *shared_exec) {
  auto exec = new exec::TrtGraphExecutor();
  exec->Init(symbol, default_ctx, group2ctx,
             in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,
             arg_shape_map, arg_dtype_map, arg_stype_map,
             grad_req_types, param_names,
             in_args, arg_grads, aux_states,
             shared_buffer, shared_exec);
  return exec;
}

}  // namespace exec

}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
