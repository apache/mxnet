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
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA

#include <cuda_runtime.h>

#include <mxnet/storage.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <new>

#include "../common/cuda_utils.h"
#include "storage_manager.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager : public AbstractManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }

  std::shared_ptr<storage::Handle> Alloc(std::size_t size, Context context) override;
  void Free(Handle* handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    size_t size = handle->size + NDEV;
    auto& reuse_pool = memory_pool_[size];
    reuse_pool.push_back(handle->dptr);
  }

  void DirectFree(Handle* handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle->dptr, handle->size);
  }

  void DirectFreeNoLock(void* ptr, std::size_t size) {
    auto err = cudaFree(ptr);
    size += NDEV;

    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }

    used_memory_ -= size;
  }

 private:
  void ReleaseAll();
  // used memory
  std::size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // number of devices
  const int NDEV = 32;
  // memory pool
  std::unordered_map<std::size_t, std::vector<void*>> memory_pool_;

  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

std::shared_ptr<storage::Handle> GPUPooledStorageManager::Alloc(std::size_t size, Context context) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size = size + NDEV;

  void* ret = nullptr;

  auto reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    std::size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100) {
      ReleaseAll();
    }

    auto e = cudaMalloc(&ret, size);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
    used_memory_ += size;
  } else {
    auto& reuse_pool = reuse_it->second;
    ret = reuse_pool.back();
    reuse_pool.pop_back();
  }

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ret;
  handle->size = size;
  handle->ctx = context;
  /* todo: need to be set */
  // handle->shared_pid = -1;
  // handle->shared_id = -1;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto& pair : memory_pool_) {
    const auto& size = pair.first - NDEV;
    for (const auto& ptr : pair.second) {
      DirectFreeNoLock(ptr, size);
    }
  }
  memory_pool_.clear();
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
