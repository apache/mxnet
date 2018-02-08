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

#if defined(_WIN32) || defined(__WINDOWS__) || defined(__WIN32__) || defined(_WIN64)

#include "cpu_shared_storage_manager.h"

#include <windows.h>
#include <process.h>
#include <unordered_map>
#include <sstream>

namespace mxnet {
namespace storage {

namespace {
std::unordered_map<void*, HANDLE> handle_map;
}

std::shared_ptr<storage::Handle> CPUSharedStorageManager::Alloc(std::size_t size, Context context) {
  CHECK_EQ(context.dev_type, Context::kCPUShared)
    << "Context for the CPUSharedStorageManager should always be Context::kCPUShared";

  const auto shared_pid = static_cast<int>(GetCurrentProcessId());
  auto shared_id = Random();

  std::stringstream stream;
  stream << shared_pid << "_" << shared_id;
  auto key = stream.str();

  return Allocate(key.c_str(), size, context);
}

void CPUSharedStorageManager::Free(storage::Handle* handle) {
  auto flag = UnmapViewOfFile(handle->dptr);
  CHECK(flag) << "Failed to unmap shared memory. UnmapViewOfFile error: " << GetLastError();

  auto it = handle_map.find(handle->dptr);
  CHECK(it != handle_map.end())
    << "Could not find allocation, freeing handle which was not mapped";

  const auto& win_handle = it->second;
  flag = CloseHandle(win_handle);
  CHECK(flag) << "Failed to close file handle. CloseHandle error: " << GetLastError();

  auto removed = handle_map.erase(it);
  CHECK_EQ(removed, 1) << "Should be 1 element removed from handle map";

  handle->dptr = nullptr;
}

std::shared_ptr<storage::Handle> CPUSharedStorageManager::Attach(const char* key) {
  auto filename = GetSharedKey(key);

  auto map_handle = OpenFileMapping(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, filename.c_str());

  auto error = GetLastError();
  if (error != ERROR_SUCCESS || !map_handle) {
    LOG(WARNING) << "Failed to open shared memory. OpenFileMapping error: " << error;
    return nullptr;
  }

  auto ptr = MapViewOfFile(map_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
  CHECK(ptr) << "Failed to map shared memory. MapViewOfFile error: " << GetLastError();

  MEMORY_BASIC_INFORMATION info;
  auto bytes = VirtualQuery(ptr, &info, sizeof(info));
  CHECK_GT(bytes, 0) << "Failed to get the size of mapped shared memory. VirtualQuery error: " <<
                     GetLastError();

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = info.RegionSize;
  handle->ctx = Context::CPUShared(0);
  handle->key = key;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

std::shared_ptr<storage::Handle>
CPUSharedStorageManager::Allocate(const char* key, std::size_t size, Context context) {
  CHECK_EQ(context.dev_type, Context::kCPUShared)
    << "Context for the CPUSharedStorageManager should always be Context::kCPUShared";

  auto filename = GetSharedKey(key);
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
  handle->key = key;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

}  // namespace storage
}  // namespace mxnet

#endif  // defined(_WIN32) || defined(__WINDOWS__) || defined(__WIN32__) || defined(_WIN64)
