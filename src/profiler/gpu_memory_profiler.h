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
#ifndef MXNET_PROFILER_GPU_MEMORY_PROFILER_H_
#define MXNET_PROFILER_GPU_MEMORY_PROFILER_H_

#include <fstream>
#include <string>
#include <mxnet/storage.h>

namespace mxnet {
namespace profiler {

class GpuMemoryProfiler {
 public:
  /// \brief Get the global instance of the memory profiler
  ///        to record an allocation entry.
  static GpuMemoryProfiler* Get();
  GpuMemoryProfiler();
  /// \brief Set/Get the profiler scope.
  static void SetCurrentScope(const std::string& scope);
  static std::string GetCurrentScope();
  /// \brief Record an allocation entry in the profiler.
  void addEntry(const Storage::Handle& handle,
      const size_t actual_alloc_size, const bool reuse);
private:
  bool enabled_;
  std::ofstream alloc_csv_fout_;
};

}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_PROFILER_GPU_MEMORY_PROFILER_H_
