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
 */
#include <mxnet/storage.h>
#include "./naive_storage_manager.h"
#include "./cpu_shared_storage_manager.h"
#include "./cpu_device_storage.h"

#if MXNET_USE_CUDA
#include "./pooled_storage_manager.h"
#include "./pinned_memory_storage.h"
#include "../common/cuda_utils.h"
#endif  // MXNET_USE_CUDA

#include "../common/lazy_alloc_array.h"

namespace mxnet {

class StorageImpl : public Storage, public storage::AbstractManager {
 public:
  StorageImpl() : storage_managers_() {}

  std::shared_ptr<storage::Handle> Alloc(std::size_t size, Context context) override;

  void DirectFree(std::shared_ptr<storage::Handle>* handle) override;

  void Free(storage::Handle* handle) override;

  void DirectFree(storage::Handle* handle) override;

 private:
  std::shared_ptr<storage::AbstractManager> CommonFree(storage::Handle* handle);

  static constexpr size_t kMaxNumberOfDevices = Context::kMaxDevType + 1;

#if MXNET_USE_CUDA
  static int num_gpu_device;
#endif  // MXNET_USE_CUDA

  static void ActivateDevice(Context ctx) {
    switch (ctx.dev_type) {
      case Context::kCPU:
        break;
      case Context::kCPUShared: {
#ifdef __ANDROID__
        LOG(FATAL) << "Unimplemented device";
#endif
      }
        break;
      case Context::kGPU:
      case Context::kCPUPinned: {
#if MXNET_USE_CUDA
        if (num_gpu_device > 0) {
          CUDA_CALL(cudaSetDevice(ctx.real_dev_id()));
        }
#endif  // MXNET_USE_CUDA
        break;
      }
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }

  // internal storage managers
  std::array<
    common::LazyAllocArray<storage::AbstractManager>,
    kMaxNumberOfDevices> storage_managers_;
};  // struct Storage::Impl

#if MXNET_USE_CUDA
int StorageImpl::num_gpu_device = 0;
#endif  // MXNET_USE_CUDA

std::shared_ptr<storage::Handle> StorageImpl::Alloc(std::size_t size, Context context) {
  // space already recycled, ignore request
  auto& device = storage_managers_.at(context.dev_type);
  auto manager = device.Get(
    context.real_dev_id(), [&]() -> std::shared_ptr<storage::AbstractManager> {
      switch (context.dev_type) {
        case Context::kCPU:
          return storage::AbstractManager::make<
            storage::NaiveStorageManager<storage::CPUDeviceStorage>>();
        case Context::kCPUShared:
          return storage::AbstractManager::make<storage::CPUSharedStorageManager>();
        case Context::kCPUPinned: {
#if MXNET_USE_CUDA
          num_gpu_device = 0;
          cudaError_t e = cudaGetDeviceCount(&num_gpu_device);
          if (e != cudaSuccess) {
            num_gpu_device = 0;
          }
          if (num_gpu_device > 0) {
            return storage::AbstractManager::make<
              storage::NaiveStorageManager<storage::PinnedMemoryStorage>>();
          } else {
            return storage::AbstractManager::make<
              storage::NaiveStorageManager<storage::CPUDeviceStorage>>();
          }
#else
          return storage::AbstractManager::make<
            storage::NaiveStorageManager<storage::CPUDeviceStorage>>();
#endif  // MXNET_USE_CUDA
          break;
        }
        case Context::kGPU: {
#if MXNET_USE_CUDA
          CUDA_CALL(cudaGetDeviceCount(&num_gpu_device));
          CHECK_GT(num_gpu_device, 0) << "GPU usage requires at least 1 GPU";
          return storage::AbstractManager::make<storage::GPUPooledStorageManager>();
#else
          LOG(FATAL) << "Compile with USE_CUDA=1 to enable GPU usage";
#endif  // MXNET_USE_CUDA
          break;
        }
        default: {
          LOG(FATAL) << "Unimplemented device " << context.dev_type;
        }
      }

      return nullptr;
    });

  ActivateDevice(context);
  return manager->Alloc(size, context);
}

void StorageImpl::Free(storage::Handle* handle) {
  auto manager = CommonFree(handle);
  manager->Free(handle);
}

std::shared_ptr<storage::AbstractManager> StorageImpl::CommonFree(storage::Handle* handle) {
  const auto& context = handle->ctx;
  auto& device = storage_managers_.at(context.dev_type);
  auto manager = device.Get(context.real_dev_id(), []() {
    LOG(FATAL) << "Cannot Free space to a device you have not allocated";
    return nullptr;
  });

  ActivateDevice(context);
  return manager;
}

void StorageImpl::DirectFree(std::shared_ptr<storage::Handle>* handle) {
  storage::AbstractManager::DirectFree(handle);
}

void StorageImpl::DirectFree(storage::Handle* handle) {
  auto manager = CommonFree(handle);
  manager->DirectFree(handle);
}

std::shared_ptr<Storage> Storage::_GetSharedRef() {
#ifdef __MXNET_JS__
  // dummy code needed for emscripten code to pass
  // do not know why, the new will be NULLPTR
  static int *q = new int();
#endif
  static std::shared_ptr<Storage> inst(new StorageImpl());
  return inst;
}

Storage* Storage::Get() {
  static Storage* ptr = _GetSharedRef().get();
  return ptr;
}

}  // namespace mxnet
