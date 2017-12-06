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
 * \file cpu_device_storage.h
 * \brief CPU storage with pinned memory
 */
#ifndef MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
#define MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
#if MXNET_USE_CUDA

#include <dmlc/logging.h>
#include "mxnet/base.h"
#include "mxnet/storage.h"
#include "../common/cuda_utils.h"

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
#if MXNET_USE_NCCL
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
#endif
  // make the memory available across all devices
  CUDA_CALL(cudaHostAlloc(&ret, size, cudaHostAllocPortable));
  return ret;
}

inline void PinnedMemoryStorage::Free(void* ptr) {
#if MXNET_USE_NCCL
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
#endif
  cudaError_t err = cudaFreeHost(ptr);
  // ignore unloading error, as memory has already been recycled
  if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
    LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
  }
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_STORAGE_PINNED_MEMORY_STORAGE_H_
