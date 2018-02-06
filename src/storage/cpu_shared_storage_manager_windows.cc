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

#ifdef _WIN32

#include <Windows.h>
#include <process.h>
#include <unordered_map>

#include "cpu_shared_storage_manager.h"

namespace mxnet {
namespace storage {

namespace {
std::unordered_map<void*, HANDLE> handle_map;
}

std::shared_ptr<storage::Handle> CPUSharedStorageManager::Alloc(std::size_t size, Context context) {
  int id = -1;
  const auto shared_pid = static_cast<int>(GetCurrentProcessId());
  auto shared_id = Random();

  auto filename = SharedHandleToString(shared_pid, shared_id);
  auto map_handle = CreateFileMapping(INVALID_HANDLE_VALUE,
                                      NULL,
                                      PAGE_READWRITE,
                                      0,
                                      size,
                                      filename.c_str());
  auto error = GetLastError();
  if (error != ERROR_SUCCESS || !map_handle) {
    LOG(FATAL) << "Failed to open shared memory. CreateFileMapping error: " << error;
  }

  auto ptr = MapViewOfFile(map_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
  error = GetLastError();
  CHECK(ptr) << "Failed to map shared memory. MapViewOfFile error: " << error;

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = size;
  handle->ctx = context;
  handle->shared_id = shared_id;
  handle->shared_pid = shared_pid;

  handle_map[ptr] = map_handle;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

std::shared_ptr<storage::Handle>
CPUSharedStorageManager::GetByID(int shared_pid, int shared_id, std::size_t size) {
  auto filename = SharedHandleToString(shared_pid, shared_id);
  auto map_handle = OpenFileMapping(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, filename.c_str());

  auto error = GetLastError();
  if (error != ERROR_SUCCESS || !map_handle) {
    LOG(FATAL) << "Failed to open shared memory. OpenFileMapping error: " << error;
  }

  auto ptr = MapViewOfFile(map_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
  CHECK(ptr) << "Failed to map shared memory. MapViewOfFile error: " << GetLastError();

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = size;
  handle->ctx = Context::CPUShared(0);
  handle->shared_id = shared_id;
  handle->shared_pid = shared_pid;

  handle_map[ptr] = map_handle;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

void CPUSharedStorageManager::Free(storage::Handle& handle) {
  if (handle.shared_pid == -1 || handle.shared_id == 1) {
    return;
  }

  auto flag = UnmapViewOfFile(handle.dptr);
  CHECK(flag) << "Failed to unmap shared memory. UnmapViewOfFile error: " << GetLastError();

  auto it = handle_map.find(handle.dptr);
  CHECK_NE(it, handle_map.end()) << "Could not find allocation, freeing handle which was not mapped";

  auto handle = it->second;
  flag = CloseHandle(handle);
  CHECK(flag) << "Failed to close file handle. CloseHandle error: " << GetLastError();

  auto removed = handle_map.erase(it);
  CHECK_EQ(removed, 1) << "Should be 1 element removed from handle map";
}

}  // namespace storage
}  // namespace mxnet

#endif  // _WIN32
