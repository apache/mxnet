/*!
 * Copyright (c) 2015 by Contributors
 * \file gpu_storage.h
 * \brief GPU storage implementation.
 */
#ifndef MXNET_STORAGE_GPU_STORAGE_H_
#define MXNET_STORAGE_GPU_STORAGE_H_

#include "mxnet/base.h"

namespace mxnet {
namespace storage {

/*!
 * \brief GPU storage implementation.
 */
class GpuStorage {
 public:
  /*!
   * \brief Allocation.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  static void Free(void* ptr);
};  // class GpuStorage

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_STORAGE_H_
