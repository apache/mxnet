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
#include "./storage_manager.h"
#include "./naive_storage_manager.h"
#include "./pooled_storage_manager.h"
#include "./cpu_shared_storage_manager.h"
#include "./cpu_device_storage.h"
#include "./gpu_device_storage.h"
#include "./pinned_memory_storage.h"
#include "../common/lazy_alloc_array.h"
#include "../profiler/storage_profiler.h"

namespace mxnet {
namespace storage {

// consider change storage as a pure abstract class
class StorageImpl : public Storage {
 public:
  void Alloc(Handle* handle) override;
  void Free(Handle handle) override;
  void DirectFree(Handle handle) override;
  void ReleaseAll(Context ctx) override;
  void SharedIncrementRefCount(Handle handle) override;
  StorageImpl() = default;
  virtual ~StorageImpl() = default;

 private:
  std::shared_ptr<StorageManager> storage_manager(const Context &ctx) {
    auto &&device = storage_managers_.at(ctx.dev_type);
    std::shared_ptr<StorageManager> manager = device.Get(
      ctx.real_dev_id(), []() {
      LOG(FATAL) << "Cannot Free space to a device you have not allocated";
      return nullptr;
      });
    return manager;
  }

  static constexpr size_t kMaxNumberOfDevices = Context::kMaxDevType + 1;
  // internal storage managers
  std::array<common::LazyAllocArray<StorageManager>, kMaxNumberOfDevices> storage_managers_;
  profiler::DeviceStorageProfiler profiler_;
  // flag which is true when Pooled Storage Manager is used
  bool pooled_ = false;
#if MXNET_USE_CUDA
  static int num_gpu_device;
  profiler::GpuDeviceStorageProfiler *pProfilerGPU_ = nullptr;
#endif  // MXNET_USE_CUDA
};  // struct Storage::Impl

#if MXNET_USE_CUDA
int StorageImpl::num_gpu_device = 0;
#endif  // MXNET_USE_CUDA

StorageManager *CreateStorageManager(const Context &ctx, const char *context, bool *pPooled) {
  const auto env_var = env_var_name(context, pool_type);
  const char *type = getenv(env_var.c_str());
  if (type == nullptr)
    type = "Naive";   // default pool

  std::string strategy = type;
  StorageManager *ptr = nullptr;
  if (strategy == "Round") {
    ptr = new PooledStorageManager<RoundPower2, VectorContainer>(ctx);
    *pPooled = true;
  } else if (strategy == "Naive") {
    ptr = new PooledStorageManager<RoundMultiple, UnorderedMapContainer>(ctx);
    *pPooled = true;
  } else if (strategy == "Unpooled") {
    if (ctx.dev_type == Context::kCPU)
      ptr = new NaiveStorageManager<CPUDeviceStorage>();
#if MXNET_USE_CUDA
    else if (ctx.dev_type == Context::kGPU)
      ptr = new NaiveStorageManager<GPUDeviceStorage>();
    else              // Context::kCPUPinned
      ptr = new NaiveStorageManager<PinnedMemoryStorage>();
#endif
  } else {
    LOG(FATAL) << "Unknown memory pool strategy specified: " << strategy << ".";
  }

  return ptr;
}

void StorageImpl::Alloc(Storage::Handle *handle) {
  // Set dptr to nullptr when handle size is 0.
  if (handle->size == 0) {
    handle->dptr = nullptr;
    return;
  }

  // space already recycled, ignore request
  auto &&device = storage_managers_.at(handle->ctx.dev_type);
  bool *pPooled = &pooled_;
#if MXNET_USE_CUDA
  profiler::GpuDeviceStorageProfiler **ppProfiler = &pProfilerGPU_;
#else
  void *ppProfiler = nullptr;
#endif
  std::shared_ptr<StorageManager> manager = device.Get(
    handle->ctx.real_dev_id(), [handle, pPooled, ppProfiler]() {
    const auto dev_type = handle->ctx.dev_type;
#if MXNET_USE_CUDA
    num_gpu_device = 0;
    *ppProfiler = nullptr;
    switch (dev_type) {
      case Context::kGPU:
      case Context::kCPUPinned:
        if (cudaGetDeviceCount(&num_gpu_device) == cudaSuccess) {
          if (num_gpu_device > 0)
            *ppProfiler = profiler::GpuDeviceStorageProfiler::Get();
        } else {
          num_gpu_device = 0;
        }
      default:
        break;
    }
    if (dev_type == Context::kGPU)
      CHECK_GT(num_gpu_device, 0) << "GPU usage requires at least 1 GPU";
#else
    if (dev_type == Context::kGPU)
      LOG(FATAL) << "Compile with USE_CUDA=1 to enable GPU usage";
#endif

#if defined(ANDROID) || defined(__ANDROID__)
      if (dev_type == Context::kCPUShared)
        LOG(FATAL) << "Unimplemented device";
#endif

    const char *context = nullptr;
    switch (dev_type) {
      case Context::kCPU:
        context = "CPU";
        break;
      case Context::kGPU:
        context = "GPU";
        break;
      case Context::kCPUPinned:
        context = "CPU_PINNED";
        break;
      case Context::kCPUShared:
        // We will not generate the log messages for CPUShared
        // It could be as many of them as the number of "workers".
        break;
      default:
        LOG(FATAL) << "Unimplemented device " << dev_type;
    }

    StorageManager *ptr = nullptr;

    // By default, the Pooled Storage Manager will be used, if it is available
    int naive_storage_manager = dmlc::GetEnv("MXNET_USE_NAIVE_STORAGE_MANAGER", 0);
    if (!naive_storage_manager) {
      // Because, the pooled storage managers are NOT implemented yet for
      // following dev_type's, we will also use the naive storage managers
      switch (dev_type) {
#if MXNET_USE_CUDA
        case Context::kCPUPinned: if (num_gpu_device > 0)
                                      break;
#endif
        case Context::kCPUShared:  naive_storage_manager = true;
        default:                   break;
      }
    }

    const char *storage_manager_type;
    if (naive_storage_manager) {
      storage_manager_type = "Naive";
      switch (dev_type) {
        case Context::kCPUPinned:
#if MXNET_USE_CUDA
              if (num_gpu_device > 0) {
                ptr = new NaiveStorageManager<PinnedMemoryStorage>();
                break;
              }
#endif
              context = "CPU";
        case Context::kCPU:
              ptr = new NaiveStorageManager<CPUDeviceStorage>();
              break;
#if MXNET_USE_CUDA
        case Context::kGPU:
              ptr = new NaiveStorageManager<GPUDeviceStorage>();
              break;
#endif
        case Context::kCPUShared:
              ptr = new CPUSharedStorageManager();
        default: break;
      }
    } else {
      // Some Pooled Storage Manager will be used
      storage_manager_type = "Pooled";
      ptr = CreateStorageManager(handle->ctx, context, pPooled);
    }

    if (context)
      LOG(INFO) << "Using " << storage_manager_type << " StorageManager for " << context;

    return ptr;
  });

  manager->Alloc(handle);
  profiler_.OnAlloc(*handle);
#if MXNET_USE_CUDA
  // record the allocation event in the memory profiler
  if (pProfilerGPU_ && !pooled_)
    pProfilerGPU_->OnAlloc(*handle, handle->size, false);
#endif
}

void StorageImpl::Free(Storage::Handle handle) {
  // Do nothing if dtpr is nullptr because the handle may have already
  // been freed or have not been allocated memory yet.
  if (handle.dptr == nullptr) return;

  storage_manager(handle.ctx)->Free(handle);
  profiler_.OnFree(handle);
#if MXNET_USE_CUDA
  if (pProfilerGPU_)
    pProfilerGPU_->OnFree(handle);
#endif
}

void StorageImpl::DirectFree(Storage::Handle handle) {
  // Do nothing if dtpr is nullptr because the handle may have already
  // been freed or have not been allocated memory yet.
  if (handle.dptr == nullptr) return;

  storage_manager(handle.ctx)->DirectFree(handle);
  profiler_.OnFree(handle);
#if MXNET_USE_CUDA
  if (pProfilerGPU_)
    pProfilerGPU_->OnFree(handle);
#endif
}

void StorageImpl::ReleaseAll(Context ctx) {
  storage_manager(ctx)->ReleaseAll();
}

void StorageImpl::SharedIncrementRefCount(Storage::Handle handle) {
  CHECK_EQ(handle.ctx.dev_type, Context::kCPUShared);
  auto&& device = storage_managers_.at(Context::kCPUShared);
  auto manager = device.Get(0, []() {
      LOG(FATAL) << "Cannot increment ref count before allocating any shared memory.";
      return nullptr;
    });
#if defined(ANDROID) || defined(__ANDROID__)
  LOG(FATAL) << "Shared memory not implemented on Android";
#else
  dynamic_cast<CPUSharedStorageManager*>(manager.get())->IncrementRefCount(handle);
#endif  // defined(ANDROID) || defined(__ANDROID__)
}

const std::string env_var_name(const char* dev_type, env_var_type type) {
  static const std::array<std::string, 5> name = {
                        "MEM_POOL_TYPE",
                        "POOL_PAGE_SIZE",
                        "MEM_LARGE_ALLOC_ROUND_SIZE",
                        "MEM_POOL_ROUND_LINEAR_CUTOFF",
                        "MEM_POOL_RESERVE",
                        };

  return std::string("MXNET_") + dev_type + "_" + name[type];
}

}  // namespace storage

std::shared_ptr<Storage> Storage::_GetSharedRef() {
#ifdef __MXNET_JS__
  // dummy code needed for emscripten code to pass
  // do not know why, the new will be NULLPTR
  static int *q = new int();
#endif
  static std::shared_ptr<Storage> inst(new storage::StorageImpl());
  return inst;
}

Storage* Storage::Get() {
  static Storage *ptr = _GetSharedRef().get();
  return ptr;
}
}  // namespace mxnet
