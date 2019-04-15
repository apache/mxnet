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
#ifndef MXNET_PROFILER_STORAGE_PROFILER_H_
#define MXNET_PROFILER_STORAGE_PROFILER_H_

#include <mxnet/storage.h>
#include <string>
#include <vector>
#include "./profiler.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Storage allocation/deallocation profiling via ProfileCounters
 */
class DeviceStorageProfiler {
 public:
  /*!
   * \brief Constructor
   */
  explicit DeviceStorageProfiler(const char *domain_name = "Device Storage")
    : domain_(domain_name) {
  }

  /*!
   * \brief Called when memory has been allocated in order to record the allocation size
   * \param handle Handle to the allocated storage
   */
  void OnAlloc(const Storage::Handle &handle) {
    if (handle.size > 0) {
      profiler::Profiler *prof = profiler::Profiler::Get();
      if (prof->IsProfiling(profiler::Profiler::kMemory)) {
        Init();
        const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
        CHECK_LT(idx, mem_counters_.size()) << "Invalid device index: " << idx;
        *mem_counters_[idx] += handle.size;
      }
    }
  }

  /*!
   * \brief Called when memory has been freed in order to record the deallocation size
   * \param handle Handle to the allocated storage
   */
  void OnFree(const Storage::Handle &handle) {
    if (handle.size > 0) {
      profiler::Profiler *prof = profiler::Profiler::Get();
      if (prof->IsProfiling(profiler::Profiler::kMemory)) {
        Init();  // In case of bug which tries to free first
        const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
        CHECK_LT(idx, mem_counters_.size()) << "Invalid device index: " << idx;
        if (*mem_counters_[idx] >= handle.size) {
            *mem_counters_[idx] -= handle.size;
        } else {
            *mem_counters_[idx] = 0;
        }
      }
    }
  }

 private:
  /*!
   * \brief Lazy initialization.  No locks occur except for on the first pass
   * (or colliding parallel first passes)
   */
  void Init() {
    if (mem_counters_.empty()) {
      std::unique_lock<std::mutex> lk(init_mutex_);
      // Check again in case of collision and someone else filled it
      if (mem_counters_.empty()) {
        profiler::Profiler *prof = profiler::Profiler::Get();
        const size_t device_count = prof->DeviceCount();
        mem_counters_.reserve(device_count);
        for (size_t i = 0, n = device_count; i < n; ++i) {
          std::string name = "Memory: ";
          name += prof->DeviceName(i);
          mem_counters_.emplace_back(std::make_shared<profiler::ProfileCounter>(name.c_str(),
                                                                              &domain_));
        }
      }
    }
  }

  /*! \brief Domain of the memory profiling information */
  profiler::ProfileDomain domain_;
  /*! \brief Mutex for lazy init */
  std::mutex init_mutex_;
  /*! \brief Constant-sized vector of memory profile counters */
  std::vector<std::shared_ptr<profiler::ProfileCounter>> mem_counters_;
};

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_PROFILER_STORAGE_PROFILER_H_
