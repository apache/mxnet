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
 *  Copyright (c) 2017 by Contributors
 * \file sort_op-inl.cuh
 * \brief CUDA implementations for sort_op.h
 */
#ifndef MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
#define MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if defined(_MSC_VER) && __CUDACC_VER_MAJOR__ == 8 && __CUDACC_VER_BUILD__ != 44
// Many CUDA 8 compilers other than V8.0.44 crash on Windows
#pragma warning("Potential crash on CUDA compiler detected. Switching sorting from CUB to Thrust")
#define SORT_WITH_THRUST
#else
#include <cub/device/device_radix_sort.cuh>
#undef SORT_WITH_THRUST
#endif
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace mxnet {
namespace op {

template <typename KDType, typename VDType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
SortByKeyWorkspaceSize(const size_t num_keys) {
#ifdef SORT_WITH_THRUST
  return 0;
#else
  size_t sortpairs_bytes = 0;
  cub::DeviceRadixSort::SortPairs<KDType, VDType>(NULL, sortpairs_bytes,
      NULL, NULL, NULL, NULL, num_keys);
  size_t keys_bytes = num_keys*sizeof(KDType);
  size_t values_bytes = num_keys*sizeof(VDType);
  return (keys_bytes + values_bytes + sortpairs_bytes);
#endif
}

template<typename KDType, typename VDType>
inline void SortByKey(mshadow::Tensor<gpu, 1, KDType> keys, mshadow::Tensor<gpu, 1, VDType> values,
                      bool is_ascend, mshadow::Tensor<gpu, 1, char>* workspace,
                      const int begin_bit, const int end_bit) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
#if CUDA_VERSION >= 7000
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
#ifndef SORT_WITH_THRUST
  if (workspace != NULL) {
    // Workspace given, sort using CUB
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [keys_out, values_out, temporary_storage]
    size_t keys_bytes = keys.size(0)*sizeof(KDType);
    size_t values_bytes = keys.size(0)*sizeof(VDType);
    // Get the size of internal storage (for checking purposes only)
    size_t sortpairs_bytes = 0;
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs<KDType, VDType>(NULL, sortpairs_bytes,
          NULL, NULL, NULL, NULL,
          keys.size(0), begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending<KDType, VDType>(NULL, sortpairs_bytes,
          NULL, NULL, NULL, NULL,
          keys.size(0), begin_bit, end_bit, stream);
    }
    // Check that we have enough storage
    CHECK_GE(workspace->size(0), keys_bytes + values_bytes + sortpairs_bytes);
    //
    KDType* keys_out_ptr = reinterpret_cast<KDType *>(workspace->dptr_);
    VDType* values_out_ptr = reinterpret_cast<VDType *>(workspace->dptr_ + keys_bytes);
    void* temp_storage = reinterpret_cast<void *>(workspace->dptr_ + keys_bytes + values_bytes);
    // Sort
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs(temp_storage, sortpairs_bytes,
        keys.dptr_, keys_out_ptr, values.dptr_, values_out_ptr,
        keys.size(0), begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(temp_storage, sortpairs_bytes,
        keys.dptr_, keys_out_ptr, values.dptr_, values_out_ptr,
        keys.size(0), begin_bit, end_bit, stream);
    }
    // Copy result back to [keys, values]
    mshadow::Tensor<gpu, 1, KDType> keys_out(keys_out_ptr, mshadow::Shape1(keys.size(0)),
      keys.stream_);
    mshadow::Tensor<gpu, 1, VDType> values_out(values_out_ptr, mshadow::Shape1(keys.size(0)),
      keys.stream_);
    mshadow::Copy(keys, keys_out, keys.stream_);
    mshadow::Copy(values, values_out, values.stream_);
  } else {
#endif // SORT_WITH_THRUST
    // No workspace, sort using thrust
    thrust::device_ptr<KDType> key_iter = thrust::device_pointer_cast(keys.dptr_);
    thrust::device_ptr<VDType> value_iter = thrust::device_pointer_cast(values.dptr_);
    if (is_ascend) {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + keys.size(0), value_iter, thrust::less<KDType>());
    } else {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + keys.size(0), value_iter, thrust::greater<KDType>());
    }
#ifndef SORT_WITH_THRUST
  }
#endif // SORT_WITH_THRUST
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
#else
  LOG(FATAL) << "SortByKey is only supported for CUDA version >=7.0!";
#endif
}

template<typename DType>
inline void SortByKey(mshadow::Tensor<gpu, 1, mshadow::half::half_t> keys,
  mshadow::Tensor<gpu, 1, DType> values, bool is_ascend,
  mshadow::Tensor<gpu, 1, char>* workspace, const int begin_bit, const int end_bit) {
  LOG(FATAL) << "SortByKey for half_t is not implemented!";
}

template<typename DType>
inline void SortByKey(mshadow::Tensor<gpu, 1, DType> keys,
  mshadow::Tensor<gpu, 1, mshadow::half::half_t> values, bool is_ascend,
  mshadow::Tensor<gpu, 1, char>* workspace, const int begin_bit, const int end_bit) {
  LOG(FATAL) << "SortByKey for half_t is not implemented!";
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
