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
#include <unistd.h>
#include <random>

namespace mxnet {
namespace storage {

std::shared_ptr<storage::Handle> CPUSharedStorageManager::Alloc(std::size_t size, Context context) {
  int id = -1;
  const auto shared_pid = static_cast<int>(getpid());
  auto shared_id = Random();

  for (int i = 0; i < 10; ++i) {
    auto filename = SharedHandleToString(shared_pid, shared_id);
    id = shm_open(filename.c_str(), O_EXCL | O_CREAT | O_RDWR, 0666);
    if (id != -1) break;
    shared_id = Random();
  }

  CHECK_NE(id, -1) << "Failed to open shared memory. shm_open error: " << strerror(errno);
  CHECK_EQ(ftruncate(id, size), 0);

  auto ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, id, 0);
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map shared memory. mmap error: " << strerror(errno);

  close(id);

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = size;
  handle->ctx = context;
  handle->shared_id = shared_id;
  handle->shared_pid = shared_pid;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

std::shared_ptr<storage::Handle>
CPUSharedStorageManager::GetByID(int shared_pid, int shared_id, std::size_t size) {
  auto filename = SharedHandleToString(shared_pid, shared_id);

  auto id = shm_open(filename.c_str(), O_RDWR, 0666);
  CHECK_NE(id, -1) << "Failed to open shared memory. shm_open error: " << strerror(errno);

  auto ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, id, 0);
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map shared memory. mmap error: " << strerror(errno);

  close(id);

  std::unique_ptr<storage::Handle> handle(new storage::Handle);

  handle->dptr = ptr;
  handle->size = size;
  handle->ctx = Context::CPUShared(0);
  handle->shared_id = shared_id;
  handle->shared_pid = shared_pid;

  return std::shared_ptr<storage::Handle>(handle.release(), DefaultDeleter());
}

void CPUSharedStorageManager::Free(storage::Handle* handle) {
  if (handle->shared_pid == -1 || handle->shared_id == 1) {
    return;
  }

  auto id = munmap(handle->dptr, handle->size);
  CHECK_NE(id, -1) << "Failed to unmap shared memory. munmap error: " << strerror(errno);

  auto filename = SharedHandleToString(handle->shared_pid, handle->shared_id);
  id = shm_unlink(filename.c_str());
  CHECK_EQ(id, 0) << "Failed to unlink shared memory. shm_unlink error: " << strerror(errno);
}

}  // namespace storage
}  // namespace mxnet

#endif  // defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
