/*!
 * Copyright (c) 2015 by Contributors
 * \file utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_COMMON_UTILS_H_
#define MXNET_COMMON_UTILS_H_

#if DMLC_USE_CXX11
#include <memory>
#include <vector>
#include <type_traits>
#include <utility>
#include <random>
#include <thread>
#include <algorithm>
#endif  // DMLC_USE_CXX11

#include <dmlc/logging.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace mxnet {
// forward declaration
namespace op {
template <typename xpu>
void CastStorageComputeEx(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<NDArray>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<NDArray>& outputs);
}

namespace common {

#if DMLC_USE_CXX11
/*
 * \brief Get the corresponding tensor blobs from default storage NDArrays.
 *        If any NDArray is of non-default storage, it is casted to default storage and
 *        the temporary NDArrays are stored in `temps`. When storage_fallback is false,
 *        and `MXNET_EXEC_STORAGE_FALLBACK` == 0, storage fallback is disallowed.
 * \return true if any input is casted
 */
template <typename xpu>
inline bool GetDefaultBlobs(const std::vector<NDArray>& nds,
                            std::vector<TBlob> *blobs,
                            std::vector<NDArray> *temps,
                            const OpContext& ctx,
                            bool storage_fallback = false) {
  bool casted = false;
  if (storage_fallback == false) {
    storage_fallback = dmlc::GetEnv("MXNET_EXEC_STORAGE_FALLBACK", true);
  }
  for (auto& nd : nds) {
    if (nd.storage_type() != kDefaultStorage) {
      if (storage_fallback == false) {
        LOG(FATAL) << "Storage type conversion detected during execution. "
                   << "You are probably executing an operator which "
                   << "doesn't support NDArray inputs with non-default storage.";
      }
      NDArray temp(nd.shape(), nd.ctx(), false);
      op::CastStorageComputeImpl<xpu>(ctx.get_stream<xpu>(), nd, temp);
      temps->push_back(temp);
      blobs->push_back(temp.data());
      casted = true;
    } else {
      blobs->push_back(nd.data());
    }
  }
  return casted;
}

template <typename xpu>
inline void GetOutputBlobs(const std::vector<NDArray>& nds,
                           std::vector<TBlob> *blobs) {
  for (auto& nd : nds) {
    blobs->push_back(nd.data());
  }
}

/*
 * \brief Cast the NDArrays in `src` according to the storage types of the NDArrays
 *        in `dst`. The ones with default storage in `dst` are ignored.
 *        When storage_fallback is false, and `MXNET_EXEC_STORAGE_FALLBACK` == 0,
 *        storage fallback is disallowed.
 */
template <typename xpu>
inline void CastNonDefaultStorage(const std::vector<NDArray>& dst,
                                  const std::vector<NDArray>& src,
                                  const OpContext& ctx,
                                  bool storage_fallback = false) {
  CHECK_GE(dst.size(), src.size());
  if (src.size() == 0) return;
  if (storage_fallback == false) {
    storage_fallback = dmlc::GetEnv("MXNET_EXEC_STORAGE_FALLBACK", true);
  }
  size_t src_idx = 0;
  for (size_t i = 0; i < dst.size(); i++) {
    auto stype = dst[i].storage_type();
    if (stype != kDefaultStorage) {
      if (storage_fallback == false) {
        LOG(FATAL) << "Storage type conversion detected during execution. "
                   << "You are probably executing an operator which "
                   << "doesn't support NDArray inputs with non-default storage.";
      }
      op::CastStorageComputeImpl(ctx.get_stream<xpu>(), src[src_idx++], dst[i]);
    }
  }
  CHECK_EQ(src_idx, src.size()) << "Not all src NDArrays are casted";
}

// Check if any storage type is not default storage
inline bool ContainsNonDefaultStorage(const nnvm::StorageTypeVector& vstorage) {
  for (auto& i : vstorage) {
    if (i != kUndefinedStorage && i != kDefaultStorage) return true;
  }
  return false;
}

inline bool ContainsDefaultStorage(const std::vector<NDArray>& ndarrays) {
  for (auto &nd : ndarrays) {
    if (nd.storage_type() == kDefaultStorage) {
      return true;
    }
  }
  return false;
}

inline FCompute GetFCompute(const Op* op, Context ctx) {
  static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompute>("FCompute<cpu>");
  static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompute>("FCompute<gpu>");
  if (ctx.dev_mask() == cpu::kDevMask) {
    return fcompute_cpu.get(op, nullptr);
  } else if (ctx.dev_mask() == gpu::kDevMask) {
    return fcompute_gpu.get(op, nullptr);
  }
  LOG(FATAL) << "Unknown device mask";
  return nullptr;
}

inline FComputeEx GetFComputeEx(const Op* op, Context ctx, int stype) {
  static auto& fcpu = nnvm::Op::GetAttr<FComputeEx>(FCOMP_EX_CPU);
  static auto& fgpu = nnvm::Op::GetAttr<FComputeEx>(FCOMP_EX_GPU);
  if (stype == kDefaultStorage) return nullptr;
  if (ctx.dev_mask() == cpu::kDevMask) {
    return fcpu.get(op, nullptr);
  } else if (ctx.dev_mask() == gpu::kDevMask) {
    return fgpu.get(op, nullptr);
  }
  LOG(FATAL) << "Unknown device mask";
  return nullptr;
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

#endif  // DMLC_USE_CXX11

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_UTILS_H_
