/*!
 * Copyright (c) 2015 by Contributors
 * \file cpu_storage.h
 * \brief CPU storage implementation.
 */
#ifndef MXNET_STORAGE_CPU_STORAGE_H_
#define MXNET_STORAGE_CPU_STORAGE_H_

#include "mxnet/base.h"

namespace mxnet {
namespace storage {

/*!
 * \brief CPU storage implementation.
 */
class CpuStorage {
 public:
  /*!
   * \brief Aligned allocation on CPU.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  static void Free(void* ptr);

 private:
  /*!
   * \brief Alignment of allocation.
   */
  static constexpr size_t alignment_ = 16;
};  // class CpuStorage

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_STORAGE_H_
