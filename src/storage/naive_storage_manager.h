/*!
 * Copyright (c) 2015 by Contributors
 * \file naive_storage_manager.h
 * \brief Naive storage manager.
 */
#ifndef MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_

#include "./storage_manager.h"
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
  void* Alloc(size_t size) override;
  void Free(void* ptr) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(NaiveStorageManager);
};

template <class DeviceStorage>
void* NaiveStorageManager::Alloc(size_t size) {
  return DeviceStorage::Alloc(size);
}

template <class DeviceStorage>
void NaiveStorageManager::Free(void* ptr) {
  DeviceStorage::Free(ptr);
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
