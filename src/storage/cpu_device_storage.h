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
 * \file cpu_device_storage.h
 * \brief CPU storage implementation.
 */
#ifndef MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
#define MXNET_STORAGE_CPU_DEVICE_STORAGE_H_

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
   * \param handle Handle struct.
   * \param failsafe Return a handle with a null dptr if out of memory, rather than exit.
   */
  inline static void Alloc(Storage::Handle* handle, bool failsafe = false);
  /*!
   * \brief Deallocation.
   * \param handle Handle struct.
   */
  inline static void Free(Storage::Handle handle);

 private:
  /*!
   * \brief Alignment of allocation.
   */
#if MXNET_USE_ONEDNN == 1 || MXNET_USE_INTGEMM == 1
  // DNNL requires special alignment. 64 is used by the DNNL library in
  // memory allocation.
  static constexpr size_t alignment_ = kDNNLAlign;
#else
  static constexpr size_t alignment_ = 16;
#endif
};  // class CPUDeviceStorage

inline void CPUDeviceStorage::Alloc(Storage::Handle* handle, bool /* failsafe */) {
  bool success = mxnet::common::AlignedMemAlloc(&(handle->dptr), handle->size, alignment_);
  if (!success)
    LOG(FATAL) << "Failed to allocate CPU Memory";
}

inline void CPUDeviceStorage::Free(Storage::Handle handle) {
  mxnet::common::AlignedMemFree(handle.dptr);
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_DEVICE_STORAGE_H_
