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
#include <nnvm/graph.h>
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

#include "../operator/mxnet_op.h"
#include "../ndarray/ndarray_function.h"

namespace mxnet {
namespace common {


/*! 
 * \brief IndPtr should be in non-decreasing order, start with 0
 *           and end with value greater or equal than size of indices.
 */
struct indptr_check {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, mshadow::default_real_t* out, const DType* in,
                                  const nnvm::dim_t end, const nnvm::dim_t idx_size) {
    if ((in[i+1] < in[i]) || (i == 0 && in[i] != static_cast<DType>(0)) ||
        (i == end && in[i] < static_cast<DType>(idx_size)))
          *out = kCSRIndPtrErr;
  }
};

/*!
 *  \brief Indices should be less than the number of columns.
 */
struct idx_check {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, mshadow::default_real_t* out,
                                  const DType* in, const nnvm::dim_t ncols) {
    if (in[i] >= static_cast<DType>(ncols)) *out = kCSRIdxErr;
  }
};

template<typename xpu>
void CheckFormatWrapper(const RunContext &rctx, const NDArray *input,
                        TBlob *cpu_err, const bool &full_check);

template<typename xpu>
void CheckFormatImpl(const RunContext &rctx, const NDArray *input,
                     TBlob *cpu_err, const bool &full_check) {
  using namespace op::mxnet_op;
  auto stype = input->storage_type();
  auto err = cpu_err->dptr<mshadow::default_real_t>();
  if (stype == kCSRStorage) {
    const TShape shape = input->shape();
    const TShape idx_shape = input->aux_shape(csr::kIdx);
    const TShape indptr_shape = input->aux_shape(csr::kIndPtr);
    const TShape storage_shape = input->storage_shape();
    if ((shape.ndim() != 2) ||
        (idx_shape.ndim() != 1 || indptr_shape.ndim() != 1 || storage_shape.ndim() != 1) ||
        (indptr_shape[0] != shape[0] + 1) ||
        (idx_shape[0] != storage_shape[0])) {
          *err = kCSRShapeErr;
          return;
    }
    if (full_check) {
      NDArray xpu_ret = NDArray(mshadow::Shape1(1), rctx.get_ctx());
      TBlob xpu_tmp = xpu_ret.data();
      ndarray::Eval<xpu>(kNormalErr, &xpu_tmp, rctx);
      int indptr_type = input->aux_type(csr::kIndPtr);
      MSHADOW_TYPE_SWITCH(indptr_type, IType, {
        Kernel<indptr_check, xpu>::Launch(
          rctx.get_stream<xpu>(), indptr_shape[0]-1, xpu_ret.data().dptr<mshadow::default_real_t>(),
          input->aux_data(csr::kIndPtr).dptr<IType>(),
          indptr_shape[0]-1, idx_shape[0]);
      });
      int idx_type = input->aux_type(csr::kIdx);
      MSHADOW_TYPE_SWITCH(idx_type, IType, {
        Kernel<idx_check, xpu>::Launch(
          rctx.get_stream<xpu>(), idx_shape[0], xpu_ret.data().dptr<mshadow::default_real_t>(),
          input->aux_data(csr::kIdx).dptr<IType>(), shape[1]);
      });
      ndarray::Copy<xpu, cpu>(xpu_ret.data(), cpu_err,
                              xpu_ret.ctx(), Context::CPU(), rctx);
    }
  } else if (stype == kRowSparseStorage) {
    if (input->aux_shape(rowsparse::kIdx)[0] != input->storage_shape()[0]) {
      *err = kRSPShapeErr;
    }
  }
}


template<typename xpu>
void CastStorageDispatch(const OpContext& ctx, const NDArray& input, const NDArray& output);

/*! \brief returns true if all storage types in `vstorage` are the same as target `stype`.
 *         false is returned for empty inputs.
 */
inline bool ContainsOnlyStorage(const StorageTypeVector& vstorage,
                                const NDArrayStorageType stype) {
  if (!vstorage.empty()) {
    for (const auto& i : vstorage) {
      if (i != stype) return false;
    }
    return true;
  }
  return false;
}

/*! \brief returns true if the storage types of arrays in `ndarrays`
 *         are the same as target `stype`. false is returned for empty inputs.
 */
inline bool ContainsOnlyStorage(const std::vector<NDArray>& ndarrays,
                                const NDArrayStorageType stype) {
  if (!ndarrays.empty()) {
    for (const auto& nd : ndarrays) {
      if (nd.storage_type() != stype) {
        return false;
      }
    }
    return true;
  }
  return false;
}


/*! \brief get string representation of dispatch_mode */
inline std::string dispatch_mode_string(const DispatchMode x) {
  switch (x) {
    case DispatchMode::kFCompute:
      return "fcompute";
    case DispatchMode::kFComputeEx:
      return "fcompute_ex";
    case DispatchMode::kFComputeFallback:
      return "fcompute_fallback";
    case DispatchMode::kVariable:
      return "variable";
    case DispatchMode::kUndefined:
      return "undefined";
  }
  return "unknown";
}


/*! \brief get string representation of storage_type */
inline std::string stype_string(const int x) {
  switch (x) {
    case kDefaultStorage:
      return "default";
    case kCSRStorage:
      return "csr";
    case kRowSparseStorage:
      return "row_sparse";
  }
  return "unknown";
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
