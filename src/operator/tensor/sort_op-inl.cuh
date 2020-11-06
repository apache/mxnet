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
#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if defined(_MSC_VER) && __CUDACC_VER_MAJOR__ == 8 && __CUDACC_VER_BUILD__ != 44
// Many CUDA 8 compilers other than V8.0.44 crash on Windows
#pragma warning("Potential crash on CUDA compiler detected. Switching sorting from CUB to Thrust")
#define SORT_WITH_THRUST
#else
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#undef SORT_WITH_THRUST
#endif
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace mxnet {
namespace op {
namespace cuda {

#define m_half mshadow::half::half_t

template<typename T>
struct less_half
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return static_cast<m_half>(lhs) < static_cast<m_half>(rhs);
  }
};

template<typename T>
struct greater_half
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return static_cast<m_half>(lhs) > static_cast<m_half>(rhs);
  }
};
}

#ifndef SORT_WITH_THRUST
template <typename KDType, typename VDType>
inline void WorkspaceSize4KeysAndValues(
  const size_t num_keys, size_t *pKeys_bytes, size_t *pValues_bytes) {
  const size_t alignment = std::max(sizeof(KDType), sizeof(VDType));
  *pKeys_bytes = PadBytes(num_keys * sizeof(KDType), alignment);
  *pValues_bytes = PadBytes(num_keys * sizeof(VDType), alignment);
}

template <typename KDType, typename VDType>
inline typename std::enable_if<!std::is_same<KDType, m_half>::value, size_t>::type
SortPairsWorkspaceSize(const size_t num_keys, const size_t batch_size) {
  size_t sortpairs_bytes = 0;
  if (batch_size > 1) {
    cub::DeviceSegmentedRadixSort::SortPairs<KDType, VDType>(nullptr, sortpairs_bytes,
      nullptr, nullptr, nullptr, nullptr, num_keys, batch_size, (int*) nullptr, (int*) nullptr);
  } else {
    cub::DeviceRadixSort::SortPairs<KDType, VDType>(nullptr, sortpairs_bytes,
      nullptr, nullptr, nullptr, nullptr, num_keys);
  }
  return sortpairs_bytes;
}

template <typename KDType, typename VDType>
inline typename std::enable_if<std::is_same<KDType, m_half>::value, size_t>::type
SortPairsWorkspaceSize(const size_t num_keys, const size_t batch_size) {
  size_t sortpairs_bytes = 0;
  if (batch_size > 1) {
    cub::DeviceSegmentedRadixSort::SortPairs<__half, VDType>(nullptr, sortpairs_bytes,
      nullptr, nullptr, nullptr, nullptr, num_keys, batch_size, (int*) nullptr, (int*) nullptr);
  } else {
    cub::DeviceRadixSort::SortPairs<__half, VDType>(nullptr, sortpairs_bytes,
      nullptr, nullptr, nullptr, nullptr, num_keys);
  }
  return sortpairs_bytes;
}
#endif

template <typename KDType, typename VDType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
SortByKeyWorkspaceSize(const size_t num_keys,
                       const size_t batch_size,
                       const bool keys_in_place,
                       const bool values_in_place) {
#ifdef SORT_WITH_THRUST
  return 0;
#else
  size_t ret = SortPairsWorkspaceSize<KDType, VDType>(num_keys, batch_size);
  if (keys_in_place || values_in_place) {
    size_t keys_bytes, values_bytes;
    WorkspaceSize4KeysAndValues<KDType, VDType>(num_keys, &keys_bytes, &values_bytes);
    if (keys_in_place)
      ret += keys_bytes;
    if (values_in_place)
      ret += values_bytes;
  }
  return ret;
#endif
}

template<typename DType>
inline typename std::enable_if<!std::is_same<DType, m_half>::value, DType *>::type
GetDevicePntr(mshadow::Tensor<gpu, 1, DType> *pntr, mshadow::Tensor<gpu, 1, DType> *sorted_pntr) {
  if (sorted_pntr) {
    mshadow::Copy(*sorted_pntr, *pntr, pntr->stream_);
    pntr = sorted_pntr;
  }
  return pntr->dptr_;
}

using half = ::half;
template<typename DType>
inline typename std::enable_if<std::is_same<DType, m_half>::value, half *>::type
GetDevicePntr(mshadow::Tensor<gpu, 1, DType> *pntr, mshadow::Tensor<gpu, 1, DType> *sorted_pntr) {
  if (sorted_pntr) {
    mshadow::Copy(*sorted_pntr, *pntr, pntr->stream_);
    pntr = sorted_pntr;
  }
  return reinterpret_cast<half *>(pntr->dptr_);
}

template<typename KDType, typename VDType>
inline typename std::enable_if<!(std::is_same<KDType, m_half>::value ||
                                 std::is_same<VDType, m_half>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType>* keys,
              mshadow::Tensor<gpu, 1, VDType>* values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit, const int batch_size,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  const size_t num_keys = keys->size(0);
  auto stream_ = keys->stream_;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(stream_);
#ifndef SORT_WITH_THRUST
  if (workspace) {
    // Workspace given, sort using CUB
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [keys_out, values_out, temporary_storage]
    size_t keys_bytes = 0, values_bytes = 0;
    if (!sorted_keys || !sorted_values) {
      WorkspaceSize4KeysAndValues<KDType, VDType>(num_keys, &keys_bytes, &values_bytes);
      if (sorted_keys)
        keys_bytes = 0;
      else
      if (sorted_values)
        values_bytes = 0;
    }
    const auto useBatch = batch_size > 1;
    const int num_items = num_keys;
    const int num_segments = batch_size;
    const int segment_length = (num_items + num_segments - 1) / num_segments;

    auto *d_segment_offsets = reinterpret_cast<int *>(workspace->dptr_);
    size_t start_keys = 0;
    if (useBatch) {
      const auto alignment = std::max(sizeof(KDType), sizeof(index_t));
      start_keys = PadBytes(sizeof(index_t) * (num_segments + 1), alignment);
      mxnet_op::Kernel<range_fwd, gpu>::Launch(stream_, (num_segments + 1), 1, 0,
                                               segment_length, kWriteTo, d_segment_offsets);
    }

    const auto start_values = start_keys + keys_bytes;
    auto* keys_out_ptr = sorted_keys? sorted_keys->dptr_ :
                         reinterpret_cast<KDType *>(workspace->dptr_ + start_keys);
    auto* values_out_ptr = sorted_values? sorted_values->dptr_ :
                           reinterpret_cast<VDType *>(workspace->dptr_ + start_values);
    auto* temp_storage = reinterpret_cast<void *>(workspace->dptr_ + start_values + values_bytes);

    // Get the size of internal storage (for checking purposes only)
    size_t sortpairs_bytes = 0;
    if (is_ascend) {
      if (useBatch) {
        cub::DeviceSegmentedRadixSort::SortPairs<KDType, VDType>(nullptr, sortpairs_bytes,
          nullptr, nullptr, nullptr, nullptr,
          num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
          begin_bit, end_bit, stream);
      } else {
        cub::DeviceRadixSort::SortPairs<KDType, VDType>(nullptr, sortpairs_bytes,
          nullptr, nullptr, nullptr, nullptr, num_keys, begin_bit, end_bit, stream);
      }
    } else {
      if (useBatch) {
        cub::DeviceSegmentedRadixSort::SortPairsDescending<KDType, VDType>(nullptr, sortpairs_bytes,
          nullptr, nullptr, nullptr, nullptr,
          num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
          begin_bit, end_bit, stream);
      } else {
        cub::DeviceRadixSort::SortPairsDescending<KDType, VDType>(nullptr, sortpairs_bytes,
          nullptr, nullptr, nullptr, nullptr, num_keys, begin_bit, end_bit, stream);
      }
    }

    const auto required_storage = sortpairs_bytes + keys_bytes + values_bytes;

    // Check that we have enough storage
    CHECK_GE(workspace->size(0), required_storage)
      << "Workspace given to SortByKey is too small: requested " << required_storage <<
      " B and got " << workspace->size(0) << " B.";

    // Sort
    if (is_ascend) {
      if (useBatch) {
        cub::DeviceSegmentedRadixSort::SortPairs(temp_storage, sortpairs_bytes,
          keys->dptr_, keys_out_ptr, values->dptr_, values_out_ptr,
          num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
          begin_bit, end_bit, stream);
      } else {
        cub::DeviceRadixSort::SortPairs(temp_storage, sortpairs_bytes,
          keys->dptr_, keys_out_ptr, values->dptr_, values_out_ptr,
          num_keys, begin_bit, end_bit, stream);
      }
    } else {
      if (useBatch) {
        cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage, sortpairs_bytes,
          keys->dptr_, keys_out_ptr, values->dptr_, values_out_ptr,
          num_items, num_segments, d_segment_offsets, d_segment_offsets + 1,
          begin_bit, end_bit, stream);
      } else {
        cub::DeviceRadixSort::SortPairsDescending(temp_storage, sortpairs_bytes,
          keys->dptr_, keys_out_ptr, values->dptr_, values_out_ptr,
          num_keys, begin_bit, end_bit, stream);
      }
    }
    // Copy result back to [keys, values]
    if (sorted_keys == nullptr) {
      mshadow::Tensor<gpu, 1, KDType> keys_out(keys_out_ptr, mshadow::Shape1(num_keys), stream_);
      mshadow::Copy(*keys, keys_out, stream_);
    }
    if (sorted_values == nullptr) {
      mshadow::Tensor<gpu, 1, VDType> values_out(values_out_ptr, mshadow::Shape1(num_keys), stream_);
      mshadow::Copy(*values, values_out, values->stream_);
    }
  } else {
#endif // SORT_WITH_THRUST
    const auto key_iter = thrust::device_pointer_cast(GetDevicePntr(keys, sorted_keys));
    const auto val_iter = thrust::device_pointer_cast(GetDevicePntr(values, sorted_values));
    if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, val_iter, thrust::less<KDType>());
    } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, val_iter, thrust::greater<KDType>());
    }
#ifndef SORT_WITH_THRUST
  }
#endif // SORT_WITH_THRUST
}

template<typename KDType, typename VDType>
inline typename std::enable_if<!std::is_same<KDType, m_half>::value &&
                                std::is_same<VDType, m_half>::value, void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType>* keys,
              mshadow::Tensor<gpu, 1, VDType>* values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit, const int batch_size,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  const size_t num_keys = keys->size(0);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys->stream_);
  const auto key_iter = thrust::device_pointer_cast(GetDevicePntr(keys, sorted_keys));
  const auto val_iter = thrust::device_pointer_cast(GetDevicePntr(values, sorted_values));
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, val_iter, thrust::less<KDType>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, val_iter, thrust::greater<KDType>());
  }
}

template<typename KDType, typename VDType>
inline typename std::enable_if<std::is_same<KDType, m_half>::value, void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType>* keys,
              mshadow::Tensor<gpu, 1, VDType>* values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit, const int batch_size,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  const size_t num_keys = keys->size(0);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys->stream_);
  const auto key_iter = thrust::device_pointer_cast(GetDevicePntr(keys, sorted_keys));
  const auto val_iter = thrust::device_pointer_cast(GetDevicePntr(values, sorted_values));
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, val_iter, cuda::less_half<half>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, val_iter, cuda::greater_half<half>());
  }
}

template<typename KDType, typename VDType>
inline void SortByKey(mshadow::Tensor<gpu, 1, KDType> keys, mshadow::Tensor<gpu, 1, VDType> values,
                      bool is_ascend, mshadow::Tensor<gpu, 1, char>* workspace,
                      const int begin_bit, const int end_bit, const int batch_size,
                      mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
                      mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
#if CUDA_VERSION < 9000
#if CUDA_VERSION < 7000
  LOG(FATAL) << "SortByKey is only supported for CUDA version >=7.0!";
#endif
  if (std::is_same<KDType, m_half>::value || std::is_same<VDType, m_half>::value)
    LOG(FATAL) << "SortByKey with fp16 keys and values is only supported for CUDA version >= 9.0";
#endif

  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
  SortByKeyImpl(&keys, &values, is_ascend, workspace,
                begin_bit, end_bit, batch_size, sorted_keys, sorted_values);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
