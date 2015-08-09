/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#include "./storage_manager.h"
#include "mxnet/base.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager with a memory pool.
 */
template <class DeviceStorage>
class PooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  PooledStorageManager() = default;
  /*!
   * \brief Default destructor.
   */
  ~PooledStorageManager() = default;
  void* Alloc(size_t size) override;
  void Free(void* ptr) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(PooledStorageManager);
};

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
