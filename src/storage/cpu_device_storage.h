/*!
 * Copyright (c) 2015 by Contributors
 * \file cpu_device_storage.h
 * \brief CPU storage implementation.
 */
#ifndef MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_CPU_DEVICE_STORAGE_H_

#include <dmlc/logging.h>
#include <cstdlib>
#include "mxnet/base.h"

namespace mxnet {
namespace storage {

/*!
 * \brief CPU storage implementation.
 */
class CPUDeviceStorage {
 public:
  /*!
   * \brief Aligned allocation on CPU.
   * \param size Size to allocate.
   * \return Pointer to the storage.
   */
  inline static void* Alloc(size_t size);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(void* ptr);

 private:
  /*!
   * \brief Alignment of allocation.
   */
  static constexpr size_t alignment_ = 16;
};  // class CPUDeviceStorage

inline void* CPUDeviceStorage::Alloc(size_t size) {
#if _MSC_VER
  void* ptr;
  ptr = _aligned_malloc(size, alignment_);
  return CHECK_NOTNULL(ptr);
#else
  void* ptr;
  int ret = posix_memalign(&ptr, alignment_, size);
  CHECK_EQ(ret, 0) << "Allocation failed";
  return ptr;
#endif
}

inline void CPUDeviceStorage::Free(void* ptr) {
#if _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
