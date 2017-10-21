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
inline bool SetupDefaultBlobs(const std::vector<NDArray>& src,
                              std::vector<TBlob> *blobs,
                              std::vector<NDArray> *temp_src,
                              std::vector<NDArray> *temp_dst,
                              std::unordered_map<uint32_t, uint32_t> *idx_map = nullptr) {
  bool require_cast = false;
  for (size_t i = 0; i < src.size(); i++) {
    auto& nd = src[i];
    if (nd.storage_type() != kDefaultStorage) {
      if (idx_map != nullptr) {
        (*idx_map)[i] = temp_dst->size();
      }
      NDArray temp(nd.shape(), nd.ctx(), false, nd.dtype());
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
                                   std::vector<TBlob> *input_blobs,
                                   std::vector<TBlob> *output_blobs,
                                   std::vector<NDArray> *pre_temp_src,
                                   std::vector<NDArray> *pre_temp_dst,
                                   std::vector<NDArray> *post_temp_src,
                                   std::vector<NDArray> *post_temp_dst,
                                   std::unordered_map<uint32_t, uint32_t> *in_temp_idx_map,
                                   const std::vector<uint32_t> &mutate_idx) {
  // populate input blobs
  SetupDefaultBlobs(ndinputs, input_blobs, pre_temp_src, pre_temp_dst, in_temp_idx_map);
  // populate output blobs
  SetupDefaultBlobs(ndoutputs, output_blobs, post_temp_dst, post_temp_src);
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
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_EXEC_UTILS_H_
