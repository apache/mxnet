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
#undef SORT_WITH_THRUST
#endif
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace mxnet {
namespace op {
namespace cuda {
template<typename T>
struct less_half
{
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef bool result_type;
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return static_cast<mshadow::half::half_t>(lhs) < static_cast<mshadow::half::half_t>(rhs);
  }
};

template<typename T>
struct greater_half
{
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef bool result_type;
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return static_cast<mshadow::half::half_t>(lhs) > static_cast<mshadow::half::half_t>(rhs);
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
inline typename std::enable_if<!std::is_same<KDType, mshadow::half::half_t>::value, size_t>::type
SortPairsWorkspaceSize(const size_t num_keys) {
  size_t sortpairs_bytes = 0;
  cub::DeviceRadixSort::SortPairs<KDType, VDType>(nullptr, sortpairs_bytes,
    nullptr, nullptr, nullptr, nullptr, num_keys);
  return sortpairs_bytes;
}

template <typename KDType, typename VDType>
inline typename std::enable_if<std::is_same<KDType, mshadow::half::half_t>::value, size_t>::type
SortPairsWorkspaceSize(const size_t num_keys) {
  size_t sortpairs_bytes = 0;
  cub::DeviceRadixSort::SortPairs<__half, VDType>(nullptr, sortpairs_bytes,
    nullptr, nullptr, nullptr, nullptr, num_keys);
  return sortpairs_bytes;
}
#endif

template <typename KDType, typename VDType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
SortByKeyWorkspaceSize(const size_t num_keys,
                       const bool keys_in_place,
                       const bool values_in_place) {
#ifdef SORT_WITH_THRUST
  return 0;
#else
  size_t keys_bytes, values_bytes;
  WorkspaceSize4KeysAndValues<KDType, VDType>(num_keys, &keys_bytes, &values_bytes);
  size_t ret = SortPairsWorkspaceSize<KDType, VDType>(num_keys);
  if (keys_in_place) {
    ret += keys_bytes;
  }
  if (values_in_place) {
    ret += values_bytes;
  }
  return ret;
#endif
}

template<typename KDType, typename VDType>
inline typename std::enable_if<!(std::is_same<KDType,mshadow::half::half_t>::value ||
                                 std::is_same<VDType,mshadow::half::half_t>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
#if CUDA_VERSION >= 7000
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
#ifndef SORT_WITH_THRUST
  if (workspace != nullptr) {
    // Workspace given, sort using CUB
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [keys_out, values_out, temporary_storage]
    size_t alignment = std::max(sizeof(KDType), sizeof(VDType));
    size_t keys_bytes = PadBytes(keys.size(0)*sizeof(KDType), alignment);
    size_t values_bytes = PadBytes(keys.size(0)*sizeof(VDType), alignment);
    // Get the size of internal storage (for checking purposes only)
    size_t sortpairs_bytes = 0;
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs<KDType, VDType>(nullptr, sortpairs_bytes,
          nullptr, nullptr, nullptr, nullptr,
          keys.size(0), begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending<KDType, VDType>(nullptr, sortpairs_bytes,
          nullptr, nullptr, nullptr, nullptr,
          keys.size(0), begin_bit, end_bit, stream);
    }

    size_t required_storage = sortpairs_bytes +
                              (sorted_keys == nullptr ? keys_bytes : 0) +
                              (sorted_values == nullptr ? values_bytes : 0);

    // Check that we have enough storage
    CHECK_GE(workspace->size(0), required_storage)
      << "Workspace given to SortByKey is too small: requested " << required_storage <<
      " B and got " << workspace->size(0) << " B.";

    size_t start_keys = 0;
    size_t start_values = start_keys +
                          (sorted_keys == nullptr ? keys_bytes : 0);
    size_t start_scratch = start_values +
                           (sorted_values == nullptr ? values_bytes : 0);
    KDType* keys_out_ptr = sorted_keys == nullptr ?
                           reinterpret_cast<KDType *>(workspace->dptr_ + start_keys) :
                           sorted_keys->dptr_;
    VDType* values_out_ptr = sorted_values == nullptr ?
                             reinterpret_cast<VDType *>(workspace->dptr_ + start_values) :
                             sorted_values->dptr_;

    void* temp_storage = reinterpret_cast<void *>(workspace->dptr_ + start_scratch);
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
    if (sorted_keys == nullptr) {
      mshadow::Tensor<gpu, 1, KDType> keys_out(keys_out_ptr, mshadow::Shape1(keys.size(0)),
        keys.stream_);
      mshadow::Copy(keys, keys_out, keys.stream_);
    }
    if (sorted_values == nullptr) {
      mshadow::Tensor<gpu, 1, VDType> values_out(values_out_ptr, mshadow::Shape1(keys.size(0)),
        keys.stream_);
      mshadow::Copy(values, values_out, values.stream_);
    }
  } else {
#endif // SORT_WITH_THRUST
    // No workspace, sort using thrust
    auto* k = &keys;
    auto* v = &values;
    if (sorted_keys != nullptr) {
      k = sorted_keys;
      mshadow::Copy(*sorted_keys, keys, keys.stream_);
    }
    if (sorted_values != nullptr) {
      v = sorted_values;
      mshadow::Copy(*sorted_values, values, values.stream_);
    }
    const auto key_iter = thrust::device_pointer_cast(k->dptr_);
    const auto value_iter = thrust::device_pointer_cast(v->dptr_);
    if (is_ascend) {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + keys.size(0), value_iter.get(), thrust::less<KDType>());
    } else {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + keys.size(0), value_iter.get(), thrust::greater<KDType>());
    }
#ifndef SORT_WITH_THRUST
  }
#endif // SORT_WITH_THRUST
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
#else
  LOG(FATAL) << "SortByKey is only supported for CUDA version >=7.0!";
#endif
}

template<typename KDType, typename VDType>
inline typename std::enable_if<((!std::is_same<KDType,mshadow::half::half_t>::value) &&
                                std::is_same<VDType,mshadow::half::half_t>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
#if CUDA_VERSION >= 9000
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
  auto* k = &keys;
  auto* v = &values;
  if (sorted_keys != nullptr) {
    k = sorted_keys;
    mshadow::Copy(*sorted_keys, keys, keys.stream_);
  }
  if (sorted_values != nullptr) {
    v = sorted_values;
    mshadow::Copy(*sorted_values, values, values.stream_);
  }
  const auto key_iter = thrust::device_pointer_cast(k->dptr_);
  const auto value_iter = thrust::device_pointer_cast(reinterpret_cast<__half*>(v->dptr_));
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter.get(), key_iter.get() + (keys.size(0)), value_iter.get(), thrust::less<KDType>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter.get(), key_iter.get() + (keys.size(0)), value_iter.get(), thrust::greater<KDType>());
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
#else
  LOG(FATAL) << "SortByKey with fp16 values is only supported for CUDA version >= 9.0";
#endif
}

template<typename KDType, typename VDType>
inline typename std::enable_if<(std::is_same<KDType,mshadow::half::half_t>::value &&
                                (!std::is_same<VDType,mshadow::half::half_t>::value)), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
#if CUDA_VERSION >= 9000
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
  auto* k = &keys;
  auto* v = &values;
  if (sorted_keys != nullptr) {
    k = sorted_keys;
    mshadow::Copy(*sorted_keys, keys, keys.stream_);
  }
  if (sorted_values != nullptr) {
    v = sorted_values;
    mshadow::Copy(*sorted_values, values, values.stream_);
  }
  const auto key_iter = thrust::device_pointer_cast(reinterpret_cast<__half*>(k->dptr_));
  const auto value_iter = thrust::device_pointer_cast(v->dptr_);
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + (keys.size(0)), value_iter.get(), cuda::less_half<__half>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + (keys.size(0)), value_iter.get(), cuda::greater_half<__half>());
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
#else
  LOG(FATAL) << "SortByKey with fp16 keys is only supported for CUDA version >= 9.0";
#endif
}

// use thrust sorting when keys or values are half_t
template<typename KDType, typename VDType>
inline typename std::enable_if<(std::is_same<KDType,mshadow::half::half_t>::value &&
                                std::is_same<VDType,mshadow::half::half_t>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit,
              mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
              mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
#if CUDA_VERSION >= 9000
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
  auto* k = &keys;
  auto* v = &values;
  if (sorted_keys != nullptr) {
    k = sorted_keys;
    mshadow::Copy(*sorted_keys, keys, keys.stream_);
  }
  if (sorted_values != nullptr) {
    v = sorted_values;
    mshadow::Copy(*sorted_values, values, values.stream_);
  }
  const auto key_iter = thrust::device_pointer_cast(reinterpret_cast<__half*>(k->dptr_));
  const auto value_iter = thrust::device_pointer_cast(reinterpret_cast<__half*>(v->dptr_));
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + (keys.size(0)), value_iter.get(), cuda::less_half<__half>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + (keys.size(0)), value_iter.get(), cuda::greater_half<__half>());
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
#else
  LOG(FATAL) << "SortByKey with fp16 keys and values is only supported for CUDA version >= 9.0";
#endif
}

template<typename KDType, typename VDType>
inline void SortByKey(mshadow::Tensor<gpu, 1, KDType> keys, mshadow::Tensor<gpu, 1, VDType> values,
                      bool is_ascend, mshadow::Tensor<gpu, 1, char>* workspace,
                      const int begin_bit, const int end_bit,
                      mshadow::Tensor<gpu, 1, KDType>* sorted_keys,
                      mshadow::Tensor<gpu, 1, VDType>* sorted_values) {
  SortByKeyImpl(keys, values, is_ascend, workspace, begin_bit, end_bit, sorted_keys, sorted_values);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
