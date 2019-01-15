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
 * \file cpu_device_storage.h
 * \brief CPU storage implementation.
 */
#ifndef MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_CPU_DEVICE_STORAGE_H_

#include <dmlc/logging.h>
#include <cstdlib>
#include <new>
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
  inline static void* Alloc(Storage::Handle* handle);
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   */
  inline static void Free(Storage::Handle handle);

 private:
  /*!
   * \brief Alignment of allocation.
   */
#if MXNET_USE_MKLDNN == 1
  // MKLDNN requires special alignment. 64 is used by the MKLDNN library in
  // memory allocation.
  static constexpr size_t alignment_ = kMKLDNNAlign;
#else
  static constexpr size_t alignment_ = 16;
#endif
};  // class CPUDeviceStorage

inline void* CPUDeviceStorage::Alloc(Storage::Handle* handle) {
  const size_t size = handle->size;
  void* ptr;
#if _MSC_VER
  ptr = _aligned_malloc(size, alignment_);
  if (ptr == NULL) LOG(FATAL) << "Failed to allocate CPU Memory";
#else
  int ret = posix_memalign(&ptr, alignment_, size);
  if (ret != 0) LOG(FATAL) << "Failed to allocate CPU Memory";
#endif
  return ptr;
}

inline void CPUDeviceStorage::Free(Storage::Handle handle) {
  void * ptr = handle.dptr;
#if _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
