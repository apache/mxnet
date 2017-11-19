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
 * \file storage_manager.h
 * \brief Storage manager.
 */

#ifndef MXNET_STORAGE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_STORAGE_MANAGER_H_

#include <mxnet/storage.h>
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
  virtual void Alloc(Storage::Handle* handle) = 0;
  /*!
   * \brief Deallocation.
   * \param ptr Pointer to deallocate.
   * \param size Size of the storage.
   */
  virtual void Free(Storage::Handle handle) = 0;
  /*!
   * \brief Direct de-allocation.
   * \param ptr Pointer to deallocate.
   * \param size Size of the storage.
   */
  virtual void DirectFree(Storage::Handle handle) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~StorageManager() = default;
};  // namespace StorageManager

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_STORAGE_MANAGER_H_
