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

#include <vector>
#include <string>
#include <utility>
#include "../common/utils.h"

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
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> *bufs,
                                 std::vector<TBlob> *blobs,
                                 std::vector<NDArray> *temp_src,
                                 std::vector<NDArray> *temp_dst) {
  bool require_cast = false;
  for (size_t i = 0; i < src.size(); i++) {
    auto& nd = src[i];
    bool is_default = nd.storage_type() == kDefaultStorage;
#if MXNET_USE_MKLDNN == 1
    // We have to make sure it's default storage and default layout.
    is_default = nd.IsDefaultData();
#endif
    if (!is_default) {
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

/*
 * \brief setup default-storage tblobs for input and output NDArrays.
 *        If any NDArray has non-default storage,
 *        it creates a temp NDArray with default storage and uses the temp tblob. The
 *        function also records the indices of non-default source NDArrays and the indices of
 *        their corresponding temporary NDArrays in the temp array.
 */
inline void SetupDefaultBlobsInOut(const std::vector<NDArray> &ndinputs,
                                   const std::vector<NDArray> &ndoutputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> *in_bufs,
                                   const std::vector<NDArray> *out_bufs,
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
  SetupDefaultBlobsOut(ndoutputs, req, out_bufs, output_blobs, post_temp_dst,
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


}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_EXEC_UTILS_H_

