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

#ifndef MXNET_STORAGE_CPU_SHARED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_CPU_SHARED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <Windows.h>
#include <process.h>
#endif  // _WIN32

#include <unordered_map>
#include <vector>
#include <atomic>
#include <iostream>
#include <mutex>
#include <new>
#include <string>
#include <limits>

#include "./storage_manager.h"
#include "../common/cuda_utils.h"


namespace mxnet {
namespace storage {
/*!
 * \brief Storage manager for cpu shared memory
 */
class CPUSharedStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  CPUSharedStorageManager() : rand_gen_(std::random_device()()) {}
  /*!
   * \brief Default destructor.
   */
  ~CPUSharedStorageManager() {
    for (const auto& kv : pool_) {
      FreeImpl(kv.second);
    }
#ifdef _WIN32
    CheckAndRealFree();
#endif
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override {
    pool_.erase(handle.dptr);
    FreeImpl(handle);
  }

  void DirectFree(Storage::Handle handle) override {
    Free(handle);
  }

  void IncrementRefCount(const Storage::Handle& handle) {
    std::atomic<int>* counter = reinterpret_cast<std::atomic<int>*>(
        static_cast<char*>(handle.dptr) - alignment_);
    ++(*counter);
  }

  int DecrementRefCount(const Storage::Handle& handle) {
    std::atomic<int>* counter = reinterpret_cast<std::atomic<int>*>(
        static_cast<char*>(handle.dptr) - alignment_);
    return --(*counter);
  }

 private:
  static constexpr size_t alignment_ = 16;

  std::recursive_mutex mutex_;
  std::mt19937 rand_gen_;
  std::unordered_map<void*, Storage::Handle> pool_;
#ifdef _WIN32
  std::unordered_map<void*, Storage::Handle> is_free_;
  std::unordered_map<void*, HANDLE> map_handle_map_;
#endif

  void FreeImpl(const Storage::Handle& handle);
#ifdef _WIN32
  void CheckAndRealFree();
#endif

  std::string SharedHandleToString(int shared_pid, int shared_id) {
    std::stringstream name;
    name << "/mx_" << std::hex << shared_pid << "_" << std::hex << shared_id;
    return name.str();
  }
  DISALLOW_COPY_AND_ASSIGN(CPUSharedStorageManager);
};  // class CPUSharedStorageManager

void CPUSharedStorageManager::Alloc(Storage::Handle* handle) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::uniform_int_distribution<> dis(0, std::numeric_limits<int>::max());
  int fid = -1;
  bool is_new = false;
  size_t size = handle->size + alignment_;
  void *ptr = nullptr;
  #ifdef _WIN32
  CheckAndRealFree();
  HANDLE map_handle = nullptr;
  uint32_t error = 0;
  if (handle->shared_id == -1 && handle->shared_pid == -1) {
    is_new = true;
    handle->shared_pid = _getpid();
    for (int i = 0; i < 10; ++i) {
      handle->shared_id = dis(rand_gen_);
      auto filename = SharedHandleToString(handle->shared_pid, handle->shared_id);
      map_handle = CreateFileMapping(INVALID_HANDLE_VALUE,
                                     NULL, PAGE_READWRITE, 0, size, filename.c_str());
      if ((error = GetLastError()) == ERROR_SUCCESS) {
        break;;
      }
    }
  } else {
    auto filename = SharedHandleToString(handle->shared_pid, handle->shared_id);
    map_handle = OpenFileMapping(FILE_MAP_READ | FILE_MAP_WRITE,
                                 FALSE, filename.c_str());
    error = GetLastError();
  }

  if (error != ERROR_SUCCESS && map_handle == nullptr) {
    LOG(FATAL) << "Failed to open shared memory. CreateFileMapping failed with error "
               << error;
  }

  ptr = MapViewOfFile(map_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
  CHECK_NE(ptr, (void *)0)
      << "Failed to map shared memory. MapViewOfFile failed with error " << GetLastError();
  map_handle_map_[ptr] = map_handle;
#else
  if (handle->shared_id == -1 && handle->shared_pid == -1) {
    is_new = true;
    handle->shared_pid = getpid();
    for (int i = 0; i < 10; ++i) {
      handle->shared_id = dis(rand_gen_);
      auto filename = SharedHandleToString(handle->shared_pid, handle->shared_id);
      fid = shm_open(filename.c_str(), O_EXCL|O_CREAT|O_RDWR, 0666);
      if (fid != -1) break;
    }
  } else {
    auto filename = SharedHandleToString(handle->shared_pid, handle->shared_id);
    fid = shm_open(filename.c_str(), O_RDWR, 0666);
  }

  if (fid == -1) {
    LOG(FATAL) << "Failed to open shared memory. shm_open failed with error "
               << strerror(errno);
  }

  if (is_new) CHECK_EQ(ftruncate(fid, size), 0);

  ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fid, 0);
  CHECK_NE(ptr, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  close(fid);
#endif  // _WIN32

  if (is_new) {
    new (ptr) std::atomic<int>(1);
  }
  handle->dptr = static_cast<char*>(ptr) + alignment_;
  pool_[handle->dptr] = *handle;
}

void CPUSharedStorageManager::FreeImpl(const Storage::Handle& handle) {
  int count = DecrementRefCount(handle);
  CHECK_GE(count, 0);
#ifdef _WIN32
  is_free_[handle.dptr] = handle;
#else
  CHECK_EQ(munmap(static_cast<char*>(handle.dptr) - alignment_,
                  handle.size + alignment_), 0)
      << "Failed to unmap shared memory. munmap failed with error "
      << strerror(errno);

  if (count == 0) {
    auto filename = SharedHandleToString(handle.shared_pid, handle.shared_id);
    CHECK_EQ(shm_unlink(filename.c_str()), 0)
        << "Failed to unlink shared memory. shm_unlink failed with error "
        << strerror(errno);
  }
#endif  // _WIN32
}

#ifdef _WIN32
inline void CPUSharedStorageManager::CheckAndRealFree() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto it = std::begin(is_free_); it != std::end(is_free_);) {
    void* ptr = static_cast<char*>(it->second.dptr) - alignment_;
    std::atomic<int>* counter = reinterpret_cast<std::atomic<int>*>(
      static_cast<char*>(it->second.dptr) - alignment_);
    if ((*counter) == 0) {
      CHECK_NE(UnmapViewOfFile(ptr), 0)
        << "Failed to UnmapViewOfFile shared memory ";
      CHECK_NE(CloseHandle(map_handle_map_[ptr]), 0)
        << "Failed to CloseHandle shared memory ";
      map_handle_map_.erase(ptr);
      it = is_free_.erase(it);
    } else {
      ++it;
    }
  }
}
#endif  // _WIN32
}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_SHARED_STORAGE_MANAGER_H_
