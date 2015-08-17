/*!
 * Copyright (c) 2015 by Contributors
 * \file storage_manager.h
 * \brief Storage manager.
 */
#ifndef MXNET_STORAGE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_STORAGE_MANAGER_H_

#include <cstddef>

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager interface.
 */
class StorageManager {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  virtual void* Alloc(size_t size) = 0;
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   * \param size Size of the storage.
   */
  virtual void Free(void* ptr, size_t size) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~StorageManager() = default;
};  // namespace StorageManager

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_STORAGE_MANAGER_H_
