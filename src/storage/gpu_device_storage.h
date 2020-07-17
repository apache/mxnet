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
 * \file gpu_device_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_GPU_DEVICE_STORAGE_H_

#if MXNET_USE_CUDA
#include "mxnet/storage.h"

namespace mxnet {
namespace storage {

/*!
 * \brief GPU storage implementation.
 */
class GPUDeviceStorage {
 public:
  /*!
   * \brief Allocation.
   * \param handle Handle struct.
   */
  inline static void Alloc(Storage::Handle* handle);
  /*!
   * \brief Deallocation.
   * \param handle Handle struct.
   */
  inline static void Free(Storage::Handle handle);
};  // class GPUDeviceStorage

inline void GPUDeviceStorage::Alloc(Storage::Handle* handle) {
  mxnet::common::cuda::DeviceStore device_store(handle->ctx.real_dev_id(), true);
#if MXNET_USE_NCCL
  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#endif  // MXNET_USE_NCCL
  CUDA_CALL(cudaMalloc(&handle->dptr, handle->size));
  profiler::GpuDeviceStorageProfiler::Get()->OnAlloc(*handle, handle->size, false);
}

inline void GPUDeviceStorage::Free(Storage::Handle handle) {
  mxnet::common::cuda::DeviceStore device_store(handle.ctx.real_dev_id(), true);
#if MXNET_USE_NCCL
  std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#endif  // MXNET_USE_NCCL
  CUDA_CALL(cudaFree(handle.dptr))
  profiler::GpuDeviceStorageProfiler::Get()->OnFree(handle);
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_STORAGE_GPU_DEVICE_STORAGE_H_
