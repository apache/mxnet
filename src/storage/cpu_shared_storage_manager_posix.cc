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

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))

#include "cpu_shared_storage_manager.h"

#include <sys/mman.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <random>

namespace mxnet {
namespace storage {

std::shared_ptr<storage::Handle> CPUSharedStorageManager::Alloc(std::size_t size, Context context) {
  CHECK_EQ(context.dev_type, Context::kCPUShared)
    << "Context for the CPUSharedStorageManager should always be Context::kCPUShared";

  const auto shared_pid = static_cast<int>(getpid());
  auto shared_id = Random();

  std::stringstream stream;
  stream << shared_pid << "_" << shared_id;
  auto key = stream.str();

  return Allocate(key.c_str(), size, context);
}

void CPUSharedStorageManager::Free(storage::Handle* handle) {
  auto id = munmap(handle->dptr, handle->size);
  if (id == -1) {
    LOG(WARNING) << "Failed to unmap shared memory. munmap error: " << strerror(errno);
  }

  if (handle->key.empty()) {
    LOG(WARNING) << "Shared memory key is empty, can not unlink";
    return;
  }

  auto filename = GetSharedKey(handle->key.c_str());
  id = shm_unlink(filename.c_str());
  if (id == -1) {
    LOG(WARNING) << "Failed to unlink shared memory. shm_unlink error: " << strerror(errno);
  }

  handle->dptr = nullptr;
}

std::shared_ptr<storage::Handle> CPUSharedStorageManager::Attach(const char* key) {
  auto filename = GetSharedKey(key);

  auto id = shm_open(filename.c_str(), O_RDWR, 0666);
  if (id == -1) {
    LOG(WARNING) << "Failed to open shared memory. shm_open error: " << strerror(errno);
    return nullptr;
  }

  struct stat statbuf;
  auto flag = fstat(id, &statbuf);
  CHECK_NE(id, -1) << "Failed to get shared memory size. fstat error: " << strerror(errno);

  auto size = static_cast<std::size_t>(statbuf.st_size);

  auto ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, id, 0);
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map shared memory. mmap error: " << strerror(errno);

  close(id);

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = size;
  handle->ctx = Context::CPUShared(0);
  handle->key = key;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

std::shared_ptr<storage::Handle>
CPUSharedStorageManager::Allocate(const char* key, std::size_t size, Context context) {
  CHECK_EQ(context.dev_type, Context::kCPUShared)
    << "Context for the CPUSharedStorageManager should always be Context::kCPUShared";

  auto filename = GetSharedKey(key);
  auto id = shm_open(filename.c_str(), O_EXCL | O_CREAT | O_RDWR, 0666);
  CHECK_NE(id, -1) << "Failed to open shared memory. shm_open error: " << strerror(errno);
  CHECK_EQ(ftruncate(id, size), 0);

  auto ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, id, 0);
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map shared memory. mmap error: " << strerror(errno);

  close(id);

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = size;
  handle->ctx = context;
  handle->key = key;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

}  // namespace storage
}  // namespace mxnet

#endif  // defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
