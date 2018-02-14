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

#include <dmlc/logging.h>
#include "storage_manager.h"

namespace mxnet {
namespace storage {

void AbstractManager::DirectFree(std::shared_ptr<Handle>* handle) {
  if (!handle || !(*handle)) {
    LOG(INFO) << "Nothing to be freed";
    return;
  }

  CHECK(handle->unique()) << "Direct free memory can be used only on unique references";

  DirectFree(handle->get());
  handle->reset();
}

void AbstractManager::DirectFree(Handle* handle) {
  Free(handle);
}

std::function<void(Handle*)> AbstractManager::DefaultDeleter() {
  auto that = shared_from_this();
  return [that](storage::Handle* handle) {
    that->Free(handle);
    delete handle;
  };
}

}  // namespace storage
}  // namespace mxnet
