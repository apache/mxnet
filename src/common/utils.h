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
 * \file utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_COMMON_UTILS_H_
#define MXNET_COMMON_UTILS_H_

#include <dmlc/logging.h>
#include <dmlc/omp.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/graph_attr_types.h>
#include <nnvm/graph_attr_types.h>

#include <memory>
#include <vector>
#include <type_traits>
#include <utility>
#include <random>
#include <string>
#include <thread>
#include <algorithm>
#include <functional>

namespace mxnet {
namespace common {

template<typename xpu>
void CastStorageDispatch(const OpContext& ctx, const NDArray& input, const NDArray& output);

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
 * \brief cast the NDArrays in `src` and store the result in NDArrays in `dst`.
 *        This is only used for storage fallback in executor.
 *        When storage_fallback is false, and `MXNET_EXEC_STORAGE_FALLBACK` == 0,
 *        storage fallback is disallowed.
 * \param src list of source NDArray to cast
 * \param dst list of destionation NDArray which hold the result of cast_storage operation
 * \param ctx operator context for cast_storage operation
 * \param storage_fallback whether storage_fallback is allowed. When set to false,
 *        its value depends on `MXNET_EXEC_STORAGE_FALLBACK`.
 */
template <typename xpu>
inline void CastNonDefaultStorage(const std::vector<NDArray>& src,
                                  const std::vector<NDArray>& dst,
                                  const OpContext& ctx,
                                  bool storage_fallback = false) {
  CHECK_GE(dst.size(), src.size());
  if (src.size() == 0) return;
  if (storage_fallback == false) {
    storage_fallback = dmlc::GetEnv("MXNET_EXEC_STORAGE_FALLBACK", true);
  }
  if (storage_fallback == false) {
    LOG(FATAL) << "Storage type conversion detected during execution. "
               << "You are probably executing an operator which "
               << "doesn't support NDArray inputs with non-default storage.";
  }
  for (size_t i = 0; i < src.size(); i++) {
    CastStorageDispatch<xpu>(ctx, src[i], dst[i]);
  }
}

// Check if any storage type is not default storage
inline bool ContainsNonDefaultStorage(const StorageTypeVector& vstorage) {
  for (const auto& i : vstorage) {
    if (i != kUndefinedStorage && i != kDefaultStorage) return true;
  }
  return false;
}

// Check if any NDArray in the list has default storage
inline bool ContainsDefaultStorage(const std::vector<NDArray>& ndarrays) {
  for (const auto &nd : ndarrays) {
    if (nd.storage_type() == kDefaultStorage) {
      return true;
    }
  }
  return false;
}

// heuristic to dermine number of threads per GPU
inline int GetNumThreadPerGPU() {
  // This is resource efficient option.
  return dmlc::GetEnv("MXNET_GPU_WORKER_NTHREADS", 2);
}

// heuristic to get number of matching colors.
// this decides how much parallelism we can get in each GPU.
inline int GetExecNumMatchColor() {
  // This is resource efficient option.
  int num_match_color = dmlc::GetEnv("MXNET_EXEC_NUM_TEMP", 1);
  return std::min(num_match_color, GetNumThreadPerGPU());
}

template<typename T, typename V>
V ParallelAccumulate(const T* a, const int n, V start) {
  V sum = start;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; ++i) {
    sum += a[i];
  }
  return sum;
}

/*!
 * \brief
 * Helper function for ParallelSort.
 * DO NOT call this function directly.
 * Use the interface ParallelSort instead.
 * Ref: https://github.com/dmlc/difacto/blob/master/src/common/parallel_sort.h
 */
template<typename RandomIt, typename Compare>
void ParallelSortHelper(RandomIt first, size_t len,
                        size_t grainsize, const Compare& comp) {
  if (len < grainsize) {
    std::sort(first, first+len, comp);
  } else {
    std::thread thr(ParallelSortHelper<RandomIt, Compare>, first, len/2, grainsize, comp);
    ParallelSortHelper(first+len/2, len - len/2, grainsize, comp);
    thr.join();
    std::inplace_merge(first, first+len/2, first+len, comp);
  }
}

/*!
 * \brief
 * Sort the elements in the range [first, last) into the ascending order defined by
 * the comparator comp.
 * If the length of the range [first, last) is greater than a certain threshold,
 * the range will be recursively divided into two and assign two threads
 * to sort each half range.
 * Ref: https://github.com/dmlc/difacto/blob/master/src/common/parallel_sort.h
 */
template<typename RandomIt, typename Compare>
void ParallelSort(RandomIt first, RandomIt last, size_t num_threads, Compare comp) {
  const auto num = std::distance(first, last);
  size_t grainsize = std::max(num / num_threads + 5, static_cast<size_t>(1024*16));
  ParallelSortHelper(first, num, grainsize, comp);
}

/*!
 * \brief
 * Sort the elements in the range [first, last) into ascending order.
 * The elements are compared using the default < operator.
 * If the length of the range [first, last) is greater than a certain threshold,
 * the range will be recursively divided into two and assign two threads
 * to sort each half range.
 * Ref: https://github.com/dmlc/difacto/blob/master/src/common/parallel_sort.h
 */
template<typename RandomIt>
void ParallelSort(RandomIt first, RandomIt last, size_t num_threads) {
  ParallelSort(first, last, num_threads,
               std::less<typename std::iterator_traits<RandomIt>::value_type>());
}

/*!
 * \brief Random Engine
 */
typedef std::mt19937 RANDOM_ENGINE;

/*!
 * \brief Helper functions.
 */
namespace helper {

/*!
 * \brief Helper for non-array type `T`.
 */
template <class T>
struct UniqueIf {
  /*!
   * \brief Type of `T`.
   */
  using SingleObject = std::unique_ptr<T>;
};

/*!
 * \brief Helper for an array of unknown bound `T`.
 */
template <class T>
struct UniqueIf<T[]> {
  /*!
   * \brief Type of `T`.
   */
  using UnknownBound = std::unique_ptr<T[]>;
};

/*!
 * \brief Helper for an array of known bound `T`.
 */
template <class T, size_t kSize>
struct UniqueIf<T[kSize]> {
  /*!
   * \brief Type of `T`.
   */
  using KnownBound = void;
};

}  // namespace helper

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param args List of arguments with which an instance of `T` will be
 *             constructed.
 * \return `std``::``unique_ptr` of an instance of type `T`.
 *
 * Constructs a non-array type `T`. The arguments `args` are passed to the
 * constructor of `T`. The function does not participate in the overload
 * resolution if `T` is an array type.
 */
template <class T, class... Args>
typename helper::UniqueIf<T>::SingleObject MakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param n The size of the array to construct.
 * \return `std``::``unique_ptr` of an instance of type `T`.
 *
 * Constructs an array of unknown bound `T`. The function does not participate
 * in the overload resolution unless `T` is an array of unknown bound.
 */
template <class T>
typename helper::UniqueIf<T>::UnknownBound MakeUnique(size_t n) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[n]{});
}

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param args List of arguments with which an instance of `T` will be
 *             constructed.
 *
 * Constructs an arrays of known bound is disallowed.
 */
template <class T, class... Args>
typename helper::UniqueIf<T>::KnownBound MakeUnique(Args&&... args) = delete;

template<typename FCompType>
FCompType GetFCompute(const nnvm::Op* op, const std::string& name,
                      const Context& ctx) {
  static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompType>(name + "<cpu>");
  static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompType>(name + "<gpu>");

  if (ctx.dev_mask() == cpu::kDevMask) {
    return fcompute_cpu.get(op, nullptr);
  } else if (ctx.dev_mask() == gpu::kDevMask) {
    return fcompute_gpu.get(op, nullptr);
  } else {
    LOG(FATAL) << "Unknown device mask";
    return nullptr;
  }
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_UTILS_H_
