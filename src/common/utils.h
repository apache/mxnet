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
 * Copyright (c) 2015 by Contributors
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

namespace mxnet {
namespace common {

/*!
 * \brief IndPtr should be non-negative, in non-decreasing order, start with 0
 *           and end with value equal with size of indices.
 */
struct csr_indptr_check {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const IType* indptr,
                                  const nnvm::dim_t end, const nnvm::dim_t idx_size) {
    if (indptr[i+1] < 0 || indptr[i+1] < indptr[i] ||
        (i == 0 && indptr[i] != 0) ||
        (i == end - 1 && indptr[end] != idx_size))
      *out = kCSRIndPtrErr;
  }
};

/*!
 *  \brief Indices should be non-negative, less than the number of columns
 *           and in ascending order per row.
 */
struct csr_idx_check {
  template<typename DType, typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const IType* idx,
                                  const RType* indptr, const nnvm::dim_t ncols) {
    for (RType j = indptr[i]; j < indptr[i+1]; j++) {
      if (idx[j] >= ncols || idx[j] < 0 ||
          (j < indptr[i+1] - 1 && idx[j] >= idx[j+1])) {
        *out = kCSRIdxErr;
        break;
      }
    }
  }
};

/*!
 *  \brief Indices of RSPNDArray should be non-negative,
 *           less than the size of first dimension and in ascending order
 */
struct rsp_idx_check {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const IType* idx,
                                  const nnvm::dim_t end, const nnvm::dim_t nrows) {
    if ((i < end && idx[i+1] <= idx[i])
        || idx[i] < 0 || idx[i] >= nrows)
      *out = kRSPIdxErr;
  }
};

template<typename xpu>
void CheckFormatWrapper(const RunContext &rctx, const NDArray &input,
                        const TBlob &err_cpu, const bool full_check);

/*!
 * \brief Check the validity of CSRNDArray.
 * \param rctx Execution context.
 * \param input Input NDArray of CSRStorage.
 * \param err_cpu Error number on cpu.
 * \param full_check If true, rigorous check, O(N) operations,
 *          otherwise basic check, O(1) operations.
 */
template<typename xpu>
void CheckFormatCSRImpl(const RunContext &rctx, const NDArray &input,
                        const TBlob &err_cpu, const bool full_check) {
  using namespace op::mxnet_op;
  CHECK_EQ(input.storage_type(), kCSRStorage)
          << "CheckFormatCSRImpl is for CSRNDArray";
  const TShape shape = input.shape();
  const TShape idx_shape = input.aux_shape(csr::kIdx);
  const TShape indptr_shape = input.aux_shape(csr::kIndPtr);
  const TShape storage_shape = input.storage_shape();
  if ((shape.ndim() != 2) ||
      (idx_shape.ndim() != 1 || indptr_shape.ndim() != 1 || storage_shape.ndim() != 1) ||
      (indptr_shape[0] != shape[0] + 1) ||
      (idx_shape[0] != storage_shape[0])) {
     MSHADOW_TYPE_SWITCH(err_cpu.type_flag_, DType, {
       DType* err = err_cpu.dptr<DType>();
       *err = kCSRShapeErr;
     });
     return;
  }
  if (full_check) {
    MSHADOW_TYPE_SWITCH(err_cpu.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(input.aux_type(csr::kIndPtr), RType, {
        MSHADOW_IDX_TYPE_SWITCH(input.aux_type(csr::kIdx), IType, {
          mshadow::Stream<xpu> *s = rctx.get_stream<xpu>();
          NDArray ret_xpu = NDArray(mshadow::Shape1(1),
                                    rctx.get_ctx(), false, err_cpu.type_flag_);
          TBlob val_xpu = ret_xpu.data();
          Kernel<set_to_int<kNormalErr>, xpu>::Launch(s, val_xpu.Size(), val_xpu.dptr<DType>());
          Kernel<csr_indptr_check, xpu>::Launch(s, indptr_shape[0] - 1, val_xpu.dptr<DType>(),
            input.aux_data(csr::kIndPtr).dptr<RType>(),
            indptr_shape[0] - 1, idx_shape[0]);
          // no need to check indices if indices are empty
          if (idx_shape[0] != 0) {
            Kernel<csr_idx_check, xpu>::Launch(s, indptr_shape[0] - 1, val_xpu.dptr<DType>(),
              input.aux_data(csr::kIdx).dptr<IType>(),
              input.aux_data(csr::kIndPtr).dptr<RType>(), shape[1]);
          }
          mshadow::Copy(err_cpu.get<cpu, 1, DType>(),
                        val_xpu.get<xpu, 1, DType>(s), s);
        });
      });
    });
  }
}

/*!
 * \brief Check the validity of RowSparseNDArray.
 * \param rctx Execution context.
 * \param input Input NDArray of RowSparseStorage.
 * \param err_cpu Error number on cpu.
 * \param full_check If true, rigorous check, O(N) operations,
 *          otherwise basic check, O(1) operations.
 */
template<typename xpu>
void CheckFormatRSPImpl(const RunContext &rctx, const NDArray &input,
                        const TBlob &err_cpu, const bool full_check) {
  using namespace op::mxnet_op;
  CHECK_EQ(input.storage_type(), kRowSparseStorage)
          << "CheckFormatRSPImpl is for RSPNDArray";
  const TShape idx_shape = input.aux_shape(rowsparse::kIdx);
  if (idx_shape[0] != input.storage_shape()[0]) {
    MSHADOW_TYPE_SWITCH(err_cpu.type_flag_, DType, {
      DType* err = err_cpu.dptr<DType>();
      *err = kRSPShapeErr;
    });
    return;
  }
  if (idx_shape[0] == 0) {
    return;
  }
  if (full_check) {
    MSHADOW_TYPE_SWITCH(err_cpu.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(input.aux_type(rowsparse::kIdx), IType, {
        mshadow::Stream<xpu> *s = rctx.get_stream<xpu>();
        NDArray ret_xpu = NDArray(mshadow::Shape1(1),
                                  rctx.get_ctx(), false, err_cpu.type_flag_);
        TBlob val_xpu = ret_xpu.data();
        Kernel<set_to_int<kNormalErr>, xpu>::Launch(s, val_xpu.Size(), val_xpu.dptr<DType>());

        Kernel<rsp_idx_check, xpu>::Launch(s, idx_shape[0],
          val_xpu.dptr<DType>(), input.aux_data(rowsparse::kIdx).dptr<IType>(),
          idx_shape[0] - 1, input.shape()[0]);
        mshadow::Copy(err_cpu.get<cpu, 1, DType>(),
                      val_xpu.get<xpu, 1, DType>(s), s);
      });
    });
  }
}

template<typename xpu>
void CheckFormatImpl(const RunContext &rctx, const NDArray &input,
                     const TBlob &err_cpu, const bool full_check) {
  int stype = input.storage_type();
  if (stype == kCSRStorage) {
    CheckFormatCSRImpl<xpu>(rctx, input, err_cpu, full_check);
  } else if (stype == kRowSparseStorage) {
    CheckFormatRSPImpl<xpu>(rctx, input, err_cpu, full_check);
  } else if (stype == kDefaultStorage) {
    // no-op for default storage
  } else {
    LOG(FATAL) << "Unknown storage type " << stype;
  }
}

/*! \brief Pick rows specified by user input index array from a row sparse ndarray
 *         and save them in the output sparse ndarray.
 */
template<typename xpu>
void SparseRetainOpForwardRspWrapper(mshadow::Stream<xpu> *s,
                                     const NDArray& input_nd,
                                     const TBlob& idx_data,
                                     const OpReqType req,
                                     NDArray* output_nd);

/* \brief Casts tensor storage type to the new type.
 */
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

/*! \brief returns true if all storage types in `vstorage` are the same as target `stype1`
 *         or `stype2'. Sets boolean if both found.
 *         false is returned for empty inputs.
 */
inline bool ContainsOnlyStorage(const StorageTypeVector& vstorage,
                                const NDArrayStorageType stype1,
                                const NDArrayStorageType stype2,
                                bool *has_both) {
  if (has_both) {
    *has_both = false;
  }
  if (!vstorage.empty()) {
    uint8_t has = 0;
    for (const auto i : vstorage) {
      if (i == stype1) {
        has |= 1;
      } else if (i == stype2) {
        has |= 2;
      } else {
        return false;
      }
    }
    if (has_both) {
      *has_both = has == 3;
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

/*! \brief returns true if the storage types of arrays in `ndarrays`
 *         are the same as targets `stype1` or `stype2`. false is returned for empty inputs.
 */
inline bool ContainsOnlyStorage(const std::vector<NDArray>& ndarrays,
                                const NDArrayStorageType stype1,
                                const NDArrayStorageType stype2,
                                bool *has_both) {
  if (has_both) {
    *has_both = false;
  }
  if (!ndarrays.empty()) {
    uint8_t has = 0;
    for (const auto& nd : ndarrays) {
      const NDArrayStorageType stype = nd.storage_type();
      if (stype == stype1) {
        has |= 1;
      } else if (stype == stype2) {
        has |= 2;
      } else {
        return false;
      }
    }
    if (has_both) {
      *has_both = has == 3;
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

/*! \brief get string representation of device type */
inline std::string dev_type_string(const int dev_type) {
  switch (dev_type) {
    case Context::kCPU:
      return "cpu";
    case Context::kGPU:
      return "gpu";
    case Context::kCPUPinned:
      return "cpu_pinned";
    case Context::kCPUShared:
      return "cpu_shared";
  }
  return "unknown";
}

/*! \brief get string representation of the operator stypes */
inline std::string operator_stype_string(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         const std::vector<int>& in_attrs,
                                         const std::vector<int>& out_attrs) {
  std::ostringstream os;
  os << "operator = " << attrs.op->name
     << "\ninput storage types = [";
  for (const int attr : in_attrs) {
    os << stype_string(attr) << ", ";
  }
  os << "]\n"
     << "output storage types = [";
  for (const int attr : out_attrs) {
    os << stype_string(attr) << ", ";
  }
  os << "]\n"
     << "params = {";
  for (auto kv : attrs.dict) {
    os << "\"" << kv.first << "\" : " << kv.second << ", ";
  }
  os << "}\n"
     << "context.dev_mask = " << dev_type_string(dev_mask);
  return os.str();
}

/*! \brief get string representation of the operator */
inline std::string operator_string(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  std::string result = "";
  std::vector<int> in_stypes;
  std::vector<int> out_stypes;
  in_stypes.reserve(inputs.size());
  out_stypes.reserve(outputs.size());
  auto xform = [](const NDArray arr) -> int { return arr.storage_type(); };
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_stypes), xform);
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_stypes), xform);
  result += operator_stype_string(attrs, ctx.run_ctx.ctx.dev_mask(), in_stypes, out_stypes);
  return result;
}

/*! \brief log message once. Intended for storage fallback warning messages. */
inline void LogOnce(const std::string& message) {
  typedef dmlc::ThreadLocalStore<std::unordered_set<std::string>> LogStore;
  auto log_store = LogStore::Get();
  if (log_store->find(message) == log_store->end()) {
    LOG(INFO) << message;
    log_store->insert(message);
  }
}

/*! \brief log storage fallback event
 */
inline void LogStorageFallback(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               const std::vector<int>* in_attrs,
                               const std::vector<int>* out_attrs) {
  static bool log = dmlc::GetEnv("MXNET_STORAGE_FALLBACK_LOG_VERBOSE", true);
  if (!log) return;
  const std::string op_str = operator_stype_string(attrs, dev_mask, *in_attrs, *out_attrs);
  std::ostringstream os;
  const char* warning = "\nThe operator with default storage type will be dispatched "
    "for execution. You're seeing this warning message because the operator above is unable "
    "to process the given ndarrays with specified storage types, context and parameter. "
    "Temporary dense ndarrays are generated in order to execute the operator. "
    "You can set environment variable MXNET_STORAGE_FALLBACK_LOG_VERBOSE to "
    "0 to suppress this warning.";
  os << "\nStorage type fallback detected:\n" << op_str << warning;
  LogOnce(os.str());
}

// heuristic to dermine number of threads per GPU
inline int GetNumThreadsPerGPU() {
  // This is resource efficient option.
  return dmlc::GetEnv("MXNET_GPU_WORKER_NTHREADS", 2);
}

// heuristic to get number of matching colors.
// this decides how much parallelism we can get in each GPU.
inline int GetExecNumMatchColor() {
  // This is resource efficient option.
  int num_match_color = dmlc::GetEnv("MXNET_EXEC_NUM_TEMP", 1);
  return std::min(num_match_color, GetNumThreadsPerGPU());
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
