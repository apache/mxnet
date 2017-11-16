/*!
 * Copyright (c) 2015 by Contributors
 * \file naive_storage_manager.h
 * \brief Naive storage manager.
 */
#ifndef MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_

#include "storage_manager.h"
#include "mxnet/base.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Naive storage manager.
 */
template <class DeviceStorage>
class NaiveStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  NaiveStorageManager() = default;
  /*!
   * \brief Default destructor.
   */
  ~NaiveStorageManager() = default;
  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    DeviceStorage::Free(handle.dptr);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(NaiveStorageManager);
};  // class NaiveStorageManager

template <class DeviceStorage>
void NaiveStorageManager<DeviceStorage>::Alloc(Storage::Handle* handle) {
  handle->dptr = DeviceStorage::Alloc(handle->size);
}

template <class DeviceStorage>
void NaiveStorageManager<DeviceStorage>::Free(Storage::Handle handle) {
  DeviceStorage::Free(handle.dptr);
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
