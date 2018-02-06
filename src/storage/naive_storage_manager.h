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
 * \file naive_storage_manager.h
 * \brief Naive storage manager.
 */
#ifndef MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_

#include "storage_manager.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Naive storage manager.
 */
template<class DeviceStorage>
class NaiveStorageManager : public AbstractManager {
 public:
  NaiveStorageManager() = default;

  ~NaiveStorageManager() = default;

  std::shared_ptr<Handle> Alloc(std::size_t size, Context context) override {
    auto handle = new storage::Handle;

    handle->dptr = DeviceStorage::Alloc(size);
    handle->size = size;
    handle->ctx = context;
    // handle->shared_id = shared_id;
    // handle->shared_pid = shared_pid;

    return std::shared_ptr<Handle>(handle, DefaultDeleter());
  }

  void Free(Handle* handle) override {
    DeviceStorage::Free(handle->dptr);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(NaiveStorageManager);
};  // class NaiveStorageManager

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_NAIVE_STORAGE_MANAGER_H_
