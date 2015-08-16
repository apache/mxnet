/*!
 * Copyright (c) 2015 by Contributors
 */
#include "mxnet/storage.h"
#include <mshadow/tensor.h>
#include <dmlc/logging.h>
#include <array>
#include "storage_manager.h"
#include "naive_storage_manager.h"
#include "pooled_storage_manager.h"
#include "cpu_device_storage.h"
#include "gpu_device_storage.h"
#include "../common/cuda_utils.h"
#include "../common/utils.h"

namespace mxnet {

struct Storage::Impl {
  static constexpr size_t kPoolThreshold = 4096 * 1024 * 1024ul;
  static constexpr size_t kMaxNumberOfDevices = 3;
  static constexpr size_t kMaxNumberOfDeviceIDs = 16;

  template <class DeviceStorage>
  using CurrentStorageManager =
      storage::PooledStorageManager<DeviceStorage, kPoolThreshold>;

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

  // std::unordered_map<
  //     int, std::unordered_map<int, std::unique_ptr<storage::StorageManager>>>
  //     storage_managers;
  std::array<std::array<std::unique_ptr<storage::StorageManager>,
                        kMaxNumberOfDeviceIDs>,
             kMaxNumberOfDevices> storage_managers;
};  // struct Storage::Impl

Storage::Handle Storage::Alloc(size_t size, Context ctx) {
  Handle hd;
  hd.ctx = ctx;
  auto&& device = impl_->storage_managers.at(ctx.dev_mask);
  auto&& device_id_it = device.at(ctx.dev_id);
  // Allocate device if necessary.
  if (!device_id_it) {
    switch (ctx.dev_mask) {
      case cpu::kDevMask:
        device_id_it = common::MakeUnique<
            Storage::Impl::CurrentStorageManager<storage::CPUDeviceStorage>>();
        break;
      case gpu::kDevMask:
        device_id_it = common::MakeUnique<
            Storage::Impl::CurrentStorageManager<storage::GPUDeviceStorage>>();
        break;
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }
  Impl::ActivateDevice(ctx);
  hd.dptr = device_id_it->Alloc(size);
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
