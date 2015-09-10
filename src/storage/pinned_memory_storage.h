/*!
 * Copyright (c) 2015 by Contributors
 * \file cpu_device_storage.h
 * \brief CPU storage with pinned memory
 */
#ifndef MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
#define MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_

#include <dmlc/logging.h>
#include "mxnet/base.h"
#include "../common/cuda_utils.h"
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace storage {

class PinnedMemoryStorage {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);

  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);
};

inline void* PinnedMemoryStorage::Alloc(size_t size) {
  void* ret = nullptr;
#if MXNET_USE_CUDA
  CUDA_CALL(cudaHostMalloc(&ret, size));
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
  return ret;
}

inline void PinnedMemoryStorage::Free(void* ptr) {
#if MXNET_USE_CUDA
  CUDA_CALL(cudaFreeHost(ptr));
#else   // MXNET_USE_CUDA
  LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
