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
 * \file exec_utils.h
 * \brief Common utility functions for executors.
 */
#ifndef MXNET_COMMON_EXEC_UTILS_H_
#define MXNET_COMMON_EXEC_UTILS_H_

#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../common/utils.h"
#include "../executor/exec_pass.h"

namespace mxnet {
namespace common {

/*
 * \brief setup default-storage tblobs from source NDArrays. If any source NDArray has non-default
 *        storage, it creates a temp NDArray with default storage and uses the temp tblob. The
 *        function also records the indices of non-default source NDArrays and the indices of
 *        their corresponding temporary NDArrays in the temp array.
 * \param src list of source NDArray
 * \param blobs list of tblobs to return
 * \param temp_src list of source NDArrays which requires temporary default storage representation
 * \param temp_dst list of temporary destination NDArrays for default storage representation
 * \param idx_map mapping from indices in source NDArrays to indices in temp_dst. When not set,
          indices are not recorded
 * \return true if any source NDArray need to cast storage
 */
inline bool SetupDefaultBlobsIn(const std::vector<NDArray>& src,
                                const std::vector<NDArray> *bufs,
                                std::vector<TBlob> *blobs,
                                std::vector<NDArray> *temp_src,
                                std::vector<NDArray> *temp_dst,
                                std::unordered_map<uint32_t, uint32_t> *idx_map) {
  bool require_cast = false;
  for (size_t i = 0; i < src.size(); i++) {
    auto& nd = src[i];
    bool is_default = nd.storage_type() == kDefaultStorage;
#if MXNET_USE_MKLDNN == 1
    // We have to make sure it's default storage and default layout.
    is_default = nd.IsDefaultData();
#endif
    if (!is_default) {
      (*idx_map)[i] = temp_dst->size();
      NDArray temp = bufs != nullptr ? bufs->at(i) : NDArray(nd.shape(), nd.ctx(),
                                                             true, nd.dtype());
#if MXNET_USE_MKLDNN == 1
      CHECK(temp.IsDefaultData());
#endif
      temp_src->emplace_back(nd);
      temp_dst->emplace_back(temp);
      blobs->emplace_back(temp.data());
      require_cast = true;
    } else {
      blobs->push_back(nd.data());
    }
  }
  return require_cast;
}

inline bool SetupDefaultBlobsOut(const std::vector<NDArray>& src,
                                 const std::vector<NDArray> *bufs,
                                 std::vector<OpReqType> *req,
                                 std::vector<TBlob> *blobs,
                                 std::vector<NDArray> *temp_src,
                                 std::vector<NDArray> *temp_dst) {
  bool require_cast = false;
  for (size_t i = 0; i < src.size(); i++) {
    auto& nd = src[i];
    bool is_default = nd.storage_type() == kDefaultStorage;
#if MXNET_USE_MKLDNN == 1
    if (req->at(i) == kWriteInplace && nd.IsMKLDNNData())
      // If it's write inplace and the output array doesn't use the default
      // layout, we'll generate a temporary output array below, which means
      // the input array and the output array are no longer the same array.
      // we should change the request type.
      req->at(i) = kWriteTo;
    // We have to make sure it's default storage and default layout.
    is_default = nd.IsDefaultData();
#endif
    if (!is_default) {
#if MXNET_USE_MKLDNN == 1
      NDArray temp;
      if (bufs != nullptr) {
        temp = bufs->at(i);
      } else if (kAddTo == req->at(i) && nd.IsMKLDNNData()) {
        temp = nd.Reorder2Default();
      } else if (kAddTo == req->at(i)) {
        temp = nd;
      } else {
        temp = NDArray(nd.shape(), nd.ctx(), true, nd.dtype());
      }
      CHECK(temp.IsDefaultData());
#else
      NDArray temp = bufs != nullptr ? bufs->at(i) : NDArray(nd.shape(), nd.ctx(),
          true, nd.dtype());
#endif
      temp_src->emplace_back(nd);
      temp_dst->emplace_back(temp);
      blobs->emplace_back(temp.data());
      require_cast = true;
    } else {
      blobs->push_back(nd.data());
    }
  }
  return require_cast;
}

/*
 * \brief setup default-storage tblobs for input and output NDArrays.
 *        If any NDArray has non-default storage,
 *        it creates a temp NDArray with default storage and uses the temp tblob. The
 *        function also records the indices of non-default source NDArrays and the indices of
 *        their corresponding temporary NDArrays in the temp array.
 */
inline void SetupDefaultBlobsInOut(const std::vector<NDArray> &ndinputs,
                                   const std::vector<NDArray> &ndoutputs,
                                   const std::vector<NDArray> *in_bufs,
                                   const std::vector<NDArray> *out_bufs,
                                   std::vector<OpReqType> *req,
                                   std::vector<TBlob> *input_blobs,
                                   std::vector<TBlob> *output_blobs,
                                   std::vector<NDArray> *pre_temp_src,
                                   std::vector<NDArray> *pre_temp_dst,
                                   std::vector<NDArray> *post_temp_src,
                                   std::vector<NDArray> *post_temp_dst,
                                   std::unordered_map<uint32_t, uint32_t> *in_temp_idx_map,
                                   const std::vector<uint32_t> &mutate_idx) {
  // populate input blobs
  SetupDefaultBlobsIn(ndinputs, in_bufs, input_blobs, pre_temp_src, pre_temp_dst,
                      in_temp_idx_map);
  // populate output blobs
  SetupDefaultBlobsOut(ndoutputs, out_bufs, req, output_blobs, post_temp_dst,
                       post_temp_src);
  // add mutable inputs to post temp list
  for (const auto idx : mutate_idx) {
    auto map_iter = in_temp_idx_map->find(idx);
    if (map_iter != in_temp_idx_map->end()) {
      post_temp_src->push_back(pre_temp_dst->at(map_iter->second));
      post_temp_dst->push_back(ndinputs[idx]);
    }
  }
}

/*
 * \brief cast the NDArrays in `src` and store the result in NDArrays in `dst`.
 *        This is only used for storage fallback in executor.
 * \param src list of source NDArray to cast
 * \param dst list of destionation NDArray which hold the result of cast_storage operation
 * \param ctx operator context for cast_storage operation
 */
inline void CastNonDefaultStorage(const std::vector<NDArray>& src,
                                  const std::vector<NDArray>& dst,
                                  const OpContext& ctx,
                                  const bool is_gpu) {
  CHECK_EQ(dst.size(), src.size());
  for (size_t i = 0; i < src.size(); i++) {
    if (is_gpu) {
#if MXNET_USE_CUDA
      CastStorageDispatch<gpu>(ctx, src[i], dst[i]);
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    } else {
      CastStorageDispatch<cpu>(ctx, src[i], dst[i]);
    }
  }
}

/*! \brief The default type inference function, which assigns all undefined
 *         types to the same type of one of the inputs or outputs.
 */
inline bool SameType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int def_v = -1;
  for (int v : *oattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) {
    for (int v : *iattr) {
      if (v != -1) {
        def_v = v; break;
      }
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    v = def_v;
  }
  return true;
}


/*! \brief The default storage type inference function, which assigns all undefined
 *         storage types to kDefaultStorage. If all of input and output storage types
 *         are kDefaultStorage, DispatchMode::kFCompute is assigned to dispatch_mode. Otherwise,
 *         DispatchMode::kFComputeFallback is assigned to dispatch_mode.
 */
inline bool DefaultStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *iattr,
                               std::vector<int> *oattr) {
  bool fallback = false;
  for (int& v : *oattr) {
    if (v == -1) v = kDefaultStorage;
    if (v != kDefaultStorage) fallback = true;
  }
  for (int& v : *iattr) {
    if (v == -1) v = kDefaultStorage;
    if (v != kDefaultStorage) fallback = true;
  }
  if (*dispatch_mode == DispatchMode::kUndefined) {
    if (fallback) {
      *dispatch_mode = DispatchMode::kFComputeFallback;
    } else {
      *dispatch_mode = DispatchMode::kFCompute;
    }
  }
  return true;
}

// string representation of storage id
inline std::string storage_str(int storage_id) {
  std::string str;
  if (storage_id == -1) {
    str = "var (-1)";
  } else if (storage_id == -2) {
    str = "external storage (-2)";
  } else {
    str = "group " + std::to_string(storage_id);
  }
  return str;
}

/* log the static memory plan of the graph. Example:
   node 0 var
   node 1 _copy
            input 0: [80,3,224,224] (47040 KB) -> var storage (-1)
            output 1: [80,3,224,224] (47040 KB) -> group 0
   node 2 var
   node 3 var
   node 4 var
   node 5 var
   node 6 BatchNorm
            input 1: [80,3,224,224] (47040 KB) -> group 0
            input 2: [3] (0 KB) -> var storage (-1)
            input 3: [3] (0 KB) -> var storage (-1)
            input 4: [3] (0 KB) -> var storage (-1)
            input 5: [3] (0 KB) -> var storage (-1)
            output 6: [80,3,224,224] (47040 KB) -> group 1
            output 7: [3] (0 KB) -> group 3
            output 8: [3] (0 KB) -> group 2
   ...
 */
inline void LogMemoryPlan(const nnvm::Graph& g) {
  const auto &idx = g.indexed_graph();
  const auto& vshape = g.GetAttr<nnvm::ShapeVector>("shape");
  const auto& vtype = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& vstorage = g.GetAttr<nnvm::StorageVector>("storage_id");
  // find node range
  uint32_t node_start = 0, node_end = idx.num_nodes();
  if (g.attrs.count("node_range")) {
    const auto& range = g.GetAttr<std::pair<uint32_t, uint32_t> >("node_range");
    node_start = range.first;
    node_end = range.second;
  }
  for (uint32_t nid = node_start; nid < node_end; ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      LOG(INFO) << "node " << nid << " var";
    } else {
      LOG(INFO) << "node " << nid << " " << inode.source->attrs.op->name;
      for (const auto& e : inode.inputs) {
        auto eid = idx.entry_id(e);
        size_t kilo_bytes = vshape[eid].Size() * mshadow::mshadow_sizeof(vtype[eid]) / 1024;
        LOG(INFO) << "\t\tinput " << eid << ": " << vshape[eid] << " ("
                  << kilo_bytes << " KB) -> " << storage_str(vstorage[eid]);
      }
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        size_t kilo_bytes = vshape[eid].Size() * mshadow::mshadow_sizeof(vtype[eid]) / 1024;
        LOG(INFO) << "\t\toutput " << eid << ": " << vshape[eid] << " ("
                  << kilo_bytes << " KB) -> " << storage_str(vstorage[eid]);
      }
    }
  }
}

/* log the static memory plan of the graph. Example:
    node 0 var
    node 1 _copy: fcompute
                input 0: default
                output 1: default
    node 2 var
    node 3 Convolution: fcompute
                input 1: default
                input 2: default
                output 3: default
    node 4 var
    node 5 var
    node 6 var
    node 7 var
    node 8 BatchNorm: fcompute
                input 3: default
                input 4: default
                input 5: default
                input 6: default
                input 7: default
                output 8: default
                output 9: default
                output 10: default
    ...
 */
inline void LogInferStorage(const nnvm::Graph& g) {
  const auto &idx = g.indexed_graph();
  const auto& vstorage_type = g.GetAttr<StorageTypeVector>("storage_type");
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  uint32_t node_start = 0, node_end = idx.num_nodes();
  if (g.attrs.count("node_range")) {
    const auto& range = g.GetAttr<std::pair<uint32_t, uint32_t> >("node_range");
    node_start = range.first;
    node_end = range.second;
  }
  for (uint32_t nid = node_start; nid < node_end; ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      LOG(INFO) << "node " << nid << " var";
    } else {
      LOG(INFO) << "node " << nid << " " << inode.source->attrs.op->name
                << ": " << dispatch_mode_string(dispatch_modes[nid]);
      for (const auto& e : inode.inputs) {
        auto eid = idx.entry_id(e);
        LOG(INFO) << "\t\tinput " << eid << ": " << stype_string(vstorage_type[eid]);
      }
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        LOG(INFO) << "\t\toutput " << eid << ": " << stype_string(vstorage_type[eid]);
      }
    }
  }
}

// prints a helpful message after shape inference errors in executor.
inline void HandleInferShapeError(const size_t num_forward_inputs,
                                  const nnvm::IndexedGraph& idx,
                                  const nnvm::ShapeVector& inferred_shapes) {
  int cnt = 10;
  std::ostringstream oss;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const TShape& inferred_shape = inferred_shapes[eid];
    if (inferred_shape.ndim() == 0 || inferred_shape.Size() == 0U) {
      const std::string& arg_name = idx[nid].source->attrs.name;
      oss << arg_name << ": " << inferred_shape << ", ";
      if (--cnt == 0) {
        oss << "...";
        break;
      }
    }
  }
  LOG(FATAL) << "InferShape pass cannot decide shapes for the following arguments "
                "(0s means unknown dimensions). Please consider providing them as inputs:\n"
             << oss.str();
}

// prints a helpful message after type inference errors in executor.
inline void HandleInferTypeError(const size_t num_forward_inputs,
                                 const nnvm::IndexedGraph& idx,
                                 const nnvm::DTypeVector& inferred_dtypes) {
  int cnt = 10;
  std::ostringstream oss;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const int inferred_dtype = inferred_dtypes[eid];
    if (inferred_dtype == -1) {
      const std::string& arg_name = idx[nid].source->attrs.name;
      oss << arg_name << ": " << inferred_dtype << ", ";
      if (--cnt == 0) {
        oss << "...";
        break;
      }
    }
  }
  LOG(FATAL) << "InferType pass cannot decide dtypes for the following arguments "
                "(-1 means unknown dtype). Please consider providing them as inputs:\n"
             << oss.str();
}

// prints a helpful message after storage type checking errors in executor.
inline void HandleInferStorageTypeError(const size_t num_forward_inputs,
                                        const nnvm::IndexedGraph& idx,
                                        const StorageTypeVector& inferred_stypes) {
  int cnt = 10;
  std::ostringstream oss;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const int inferred_stype = inferred_stypes[eid];
    if (inferred_stype == -1) {
      const std::string& arg_name = idx[nid].source->attrs.name;
      oss << arg_name << ": " << common::stype_string(inferred_stype) << ", ";
      if (--cnt == 0) {
        oss << "...";
        break;
      }
    }
  }
  LOG(FATAL) << "InferStorageType pass cannot decide storage type for the following arguments "
                "(-1 means unknown stype). Please consider providing them as inputs:\n"
             << oss.str();
}

/*!
 * \brief If the requested ndarray's shape size is less than
 * the corresponding shared_data_array's shape size and the
 * storage type is shareable, reuse the memory allocation
 * in shared_buffer; otherwise, create a zero ndarray.
 * Shareable storages include both default storage and row_sparse storage
 * if enable_row_sparse_sharing is `True`, otherwise default storage only.
 */
inline NDArray ReshapeOrCreate(const std::string& name,
                               const TShape& dest_arg_shape,
                               const int dest_arg_dtype,
                               const NDArrayStorageType dest_arg_stype,
                               const Context& ctx,
                               std::unordered_map<std::string, NDArray>* shared_buffer,
                               bool enable_row_sparse_sharing) {
  bool stype_shareable = dest_arg_stype == kDefaultStorage;
  if (enable_row_sparse_sharing) {
    stype_shareable = stype_shareable || dest_arg_stype == kRowSparseStorage;
  }
  auto it = shared_buffer->find(name);
  if (it != shared_buffer->end()) {
    // check if size is large enough for sharing
    bool size_shareable = it->second.shape().Size() >= dest_arg_shape.Size();
    if (size_shareable && stype_shareable) {  // memory can be reused
      CHECK_EQ(it->second.dtype(), dest_arg_dtype)
          << "Requested arg array's dtype does not match that of the reusable ndarray";
      CHECK_EQ(it->second.storage_type(), dest_arg_stype)
          << "Requested arg array's stype does not match that of the reusable ndarray";
      return it->second.Reshape(dest_arg_shape);
    } else if (stype_shareable) {
      LOG(WARNING) << "Bucketing: data " << name << " has a shape " << dest_arg_shape
                   << ", which is larger than already allocated shape " << it->second.shape()
                   << ". Need to re-allocate. Consider putting default bucket key to be "
                   << "the bucket taking the largest input for better memory sharing.";
      // size is not large enough, creating a larger one for sharing
      // the NDArrays in shared_buffer are guaranteed to be of shareable storages
      it->second = InitZeros(dest_arg_stype, dest_arg_shape, ctx, dest_arg_dtype);
      return it->second;
    } else {
      // not shareable storage
      return InitZeros(dest_arg_stype, dest_arg_shape, ctx, dest_arg_dtype);
    }
  } else {
    auto ret = InitZeros(dest_arg_stype, dest_arg_shape, ctx, dest_arg_dtype);
    if (stype_shareable) {
      shared_buffer->emplace(name, ret);
    }
    return ret;
  }  // if (it != shared_buffer->end())
}

/*!
 * \brief Assign context to the graph.
 * This is triggered by both simple_bind and bind flows.
 */
inline nnvm::Graph AssignContext(nnvm::Graph g,
                                 const Context& default_ctx,
                                 const std::map<std::string, Context>& ctx_map,
                                 const std::vector<Context>& in_arg_ctxes,
                                 const std::vector<Context>& arg_grad_ctxes,
                                 const std::vector<Context>& aux_state_ctxes,
                                 const std::vector<OpReqType>& grad_req_types,
                                 size_t num_forward_inputs,
                                 size_t num_forward_outputs) {
  const auto& idx = g.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  // default use default context.
  if (ctx_map.size() == 0) {
    g.attrs["context"] = std::make_shared<nnvm::any>(
        exec::ContextVector(idx.num_nodes(), default_ctx));
    for (const auto& x : in_arg_ctxes) {
      CHECK(x == default_ctx)
          << "Input array is in " << x << " while binding with ctx=" << default_ctx
          << ". All arguments must be in global context (" << default_ctx
          << ") unless group2ctx is specified for cross-device graph.";
    }
    for (const auto& x : arg_grad_ctxes) {
      CHECK(x == default_ctx)
          << "Gradient array is in " << x << " while binding with ctx="
          << default_ctx << ". All gradients must be in global context (" << default_ctx
          << ") unless group2ctx is specified for cross-device graph.";
    }
    return g;
  }

  // otherwise, use context assignment.
  std::map<Context, int> ctx2id;  // map ctx to device id
  std::vector<Context> ctx_list;  // index is device id
  nnvm::DeviceVector device(idx.num_nodes(), -1);  // index is node id
  nnvm::DeviceAssignMap device_map;  // map arg name to device id

  // loop through the user input ctx_map and
  // populate maps and lists
  for (auto &kv : ctx_map) {
    if (ctx2id.count(kv.second) == 0) {  // if context has no device id, create one
      ctx2id[kv.second] = static_cast<int>(ctx_list.size());  // assign device id to ctx
      ctx_list.push_back(kv.second);  // save ctx to the list
    }
    // assign device id to to the arg name with the corresponding ctx
    device_map[kv.first] = ctx2id.at(kv.second);
  }

  // loop through all the rest of input nodes not specified
  // in the ctx_map and populate maps and lists
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    Context ctx;
    if (mutable_nodes.count(nid)) {  // aux node is mutable
      CHECK_LT(aux_top, aux_state_ctxes.size());
      ctx = aux_state_ctxes[aux_top];
      ++aux_top;
    } else {  // regular input node is immutable
      CHECK_LT(arg_top, in_arg_ctxes.size());
      ctx = in_arg_ctxes[arg_top];
      ++arg_top;
    }
    if (ctx2id.count(ctx) == 0) {  // if the current ctx is not in the map of ctx and device id
      ctx2id[ctx] = static_cast<int>(ctx_list.size());  // assign the current ctx with device id
      ctx_list.push_back(ctx);  // save the current ctx in the list
    }
    device[nid] = ctx2id.at(ctx);  // assign device id to the current node
  }

  // loop through backward input nodes and populate maps and lists
  // the backward input nodes is the gradient of the loss wrt the output
  size_t arg_grad_offset = 0;
  // keep an offset into the arg_grad_ctxes vector,
  // since g.outputs exclude arg_grad whose req == null
  CHECK_GE(grad_req_types.size(), g.outputs.size() - num_forward_outputs)
      << "insufficient number of grad_reqs";
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i, ++arg_grad_offset) {
    while (grad_req_types[arg_grad_offset] == kNullOp) ++arg_grad_offset;
    const uint32_t nid = idx.outputs()[i].node_id;
    Context ctx = arg_grad_ctxes[arg_grad_offset];
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    int devid = ctx2id.at(ctx);
    if (device[nid] != -1) {
      CHECK_EQ(device[nid], devid) << "device of same output not equal to each other";
    } else {
      device[nid] = devid;
    }
  }

  g.attrs["device"] = std::make_shared<dmlc::any>(std::move(device));
  g = nnvm::pass::PlaceDevice(g, "__ctx_group__", device_map, "_CrossDeviceCopy");
  const auto& assigned_device = g.GetAttr<nnvm::DeviceVector>("device");

  exec::ContextVector vcontext;
  for (size_t i = 0; i < assigned_device.size(); ++i) {
    if (assigned_device[i] == -1) {
      vcontext.push_back(default_ctx);
    } else {
      vcontext.push_back(ctx_list[assigned_device[i]]);
    }
  }

  // after device planning, we should check again
  // if the assigned device of gradient node
  // corresponds to storage of grads
  auto &new_idx = g.indexed_graph();
  arg_grad_offset = 0;
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i, ++arg_grad_offset) {
    while (grad_req_types[arg_grad_offset] == kNullOp) ++arg_grad_offset;
    const uint32_t nid = new_idx.outputs()[i].node_id;
    Context ctx = arg_grad_ctxes[arg_grad_offset];
    CHECK(ctx == vcontext[nid])
        << "Trying to save gradient to " << ctx
        << " while its source node \"" << new_idx[nid].source->attrs.name
        << "\" computes it on " << vcontext[nid]
        << ". Check your ctx in NDArray allocation.";
  }

  g.attrs["context"] = std::make_shared<nnvm::any>(std::move(vcontext));
  return g;
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_EXEC_UTILS_H_

