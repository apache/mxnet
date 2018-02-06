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
 * \file storage.h
 * \brief Storage manager across multiple devices.
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_

#include <cstddef>
#include <memory>

#include <mxnet/base.h>

namespace mxnet {
namespace storage {

/*!
 * \brief Storage handle.
 */
struct Handle {
  /*!
   * \brief Pointer to the data.
   */
  void* dptr { nullptr };
  /*!
   * \brief Size of the storage.
   */
  std::size_t size { 0 };
  /*!
   * \brief Context information about device and ID.
   */
  Context ctx {};
  /*!
   * \brief Id for IPC shared memory
   */
  int shared_pid { -1 };
  int shared_id { -1 };
};  // class Handle

}  // namespace storage

/*!
 * \brief A storage interface.
 */
class AbstractStorage {
 public:
  /*!
   * \brief Allocate storage.
   *
   * When the usage count of the storage handle drops to 0 the corresponding memory it
   * is handling will be freed. The implementations should be responsible for doing the
   * necessary deallocations within the deleter of the returned shared_ptr.
   *
   * \param size The size to allocate
   * \param context The context to use
   *
   * \return Shared pointer to the storage handle
   */
  virtual std::shared_ptr<storage::Handle> Alloc(std::size_t size, Context context) = 0;

  /*!
   * \brief Direct de-allocation.
   *
   * When the usage count of the storage handle is 1 this method can be called to free memory
   * without special handling. In most of the cases it's the same as deleting the last reference
   * to a handle.
   *
   * The shared pointer reference is assumed to be the last one pointing to the handle. After the
   * call to this method the shared_ptr gets invalidated.
   *
   * \param handle Handle to the storage
   */
  virtual void DirectFree(std::shared_ptr<storage::Handle>* handle) = 0;

  virtual ~AbstractStorage() = default;
};  // class AbstractStorage

/*!
 * \brief Storage manager across multiple devices.
 */
class Storage : public AbstractStorage {
 public:
  /*!
   * \brief Returns mutex used by storage manager
   */
  std::mutex& GetMutex(Context::DeviceType dev) {
    if (dev == Context::kCPU) {
      return cpu_mutex_;
    } else {
      return gpu_mutex_;
    }
  }

  /*!
   * \return Storage singleton.
   */
  static Storage* Get();

  /*!
   * \brief Get shared pointer reference to storage singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  Storage to be destructed after X.
   *
   * \return A shared pointer to Storage singleton.
   */
  static std::shared_ptr<Storage> _GetSharedRef();

 private:
  std::mutex cpu_mutex_;
  std::mutex gpu_mutex_;
};  // class Storage

}  // namespace mxnet

#endif  // MXNET_STORAGE_H_
