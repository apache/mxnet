/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/storage.h"
#include <mshadow/tensor.h>
#include <dmlc/logging.h>
#include "./storage_manager.h"
#include "./naive_storage_manager.h"
#include "./pooled_storage_manager.h"
#include "./cpu_storage.h"
#include "./gpu_storage.h"
#include "mxnet/cuda_utils.h"

namespace mxnet {

struct Storage::Impl {
  static constexpr size_t kPoolThreshold = 4096 * 1024 * 1024ul;

  template <class DeviceStorage>
  using CurrentStorageManager = storage::PooledStorageManager<DeviceStorage, kPoolThreshold>;

  static void ActivateDevice(Context ctx) {
    switch (ctx.dev_mask) {
      case cpu::kDevMask:
        break;
      case gpu::kDevMask:
#if MXNET_USE_CUDA
        CUDA_CALL(cudaSetDevice(ctx.dev_id));
#else  // MXNET_USE_CUDA
        LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
        break;
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }

  std::unordered_map<
      int, std::unordered_map<int, std::unique_ptr<storage::StorageManager>>>
      storage_managers;
};  // struct Storage::Impl

Storage::Handle Storage::Alloc(size_t size, Context ctx) {
  Handle hd;
  hd.ctx = ctx;
  auto&& device = impl_->storage_managers[ctx.dev_mask];
  auto&& device_id_it = device.find(ctx.dev_id);
  // Allocate device if necessary.
  if (device_id_it == device.end()) {
    switch (ctx.dev_mask) {
      case cpu::kDevMask:
        device_id_it =
            device.emplace(std::make_pair(
                               ctx.dev_id,
                               std::unique_ptr<storage::StorageManager>{
                                   new Storage::Impl::CurrentStorageManager<
                                       storage::CpuStorage>{}})).first;
        break;
      case gpu::kDevMask:
        device_id_it =
            device.emplace(std::make_pair(
                               ctx.dev_id,
                               std::unique_ptr<storage::StorageManager>{
                                   new Storage::Impl::CurrentStorageManager<
                                       storage::GpuStorage>{}})).first;
        break;
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }
  Impl::ActivateDevice(ctx);
  hd.dptr = device_id_it->second->Alloc(size);
  hd.size = size;
  return hd;
}

void Storage::Free(Storage::Handle handle) {
  Impl::ActivateDevice(handle.ctx);
  impl_->storage_managers.at(handle.ctx.dev_mask)
      .at(handle.ctx.dev_id)
      ->Free(handle.dptr, handle.size);
}

Storage::~Storage() = default;

Storage* Storage::Get() {
  static Storage inst;
  return &inst;
}

Storage::Storage() : impl_{new Impl{}} {}

}  // namespace mxnet
