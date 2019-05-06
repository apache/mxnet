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
 * Copyright (c) 2019 by Contributors
 * \file gpu_memory_profiler.h
 * \author Bojian Zheng and Abhishek Tiwari
 * \brief top-level header for enabling GPU memory profiler
 */

#ifndef MXNET_PROFILER_GPU_MEMORY_PROFILER_H_
#define MXNET_PROFILER_GPU_MEMORY_PROFILER_H_

#include <mxnet/storage_tag.h>
#include <string>
#include <fstream>

/** Top-Level GPU Memory Profiling Macro
 * 
 *  Set the macro below to 1/0 to enable/disable GPU memory profiling.
 */
#define MXNET_USE_GPU_MEMORY_PROFILER 0

#if  MXNET_USE_GPU_MEMORY_PROFILER
#if !MXNET_ENABLE_STORAGE_TAGGING
#error "Storage tagging flag 'MXNET_ENABLE_STORAGE_TAGGING' in 'mxnet/storage.h' " \
    "must be enabled first for using the GPU memory profiler."
#endif  // !MXNET_ENABLE_STORAGE_TAGGING

namespace mxnet {
namespace profiler {

class GpuMemoryProfiler {
 private:
  bool _enabled;  ///< Even if compiled with the GPU memory profiler enabled,
                  ///    frontend programmers can still disable it using
                  ///    the environment variable `MXNET_USE_GPU_MEMORY_PROFILER`.
  std::ofstream _csv_fout;  ///< `_csv_fout` is used for storing
                            ///    GPU memory allocation entries in CSV format.
                            ///  The file location can be set using environment variable
                            ///    `MXNET_GPU_MEMORY_PROFILER_CSV_FNAME`
  std::ofstream _log_fout;  ///< `_log_fout` is used for logging
                            ///    unknown entries and their callstack.
                            ///  The file location can be set using environment variable
                            ///    `MXNET_GPU_MEMORY_PROFILER_LOG_FNAME`
 public:
  /// \brief Get the static instance of the GPU memory profiler.
  static GpuMemoryProfiler * Get();
  GpuMemoryProfiler();
  /// \brief Record a GPU memory allocation entry in the memory profiler.
  void addEntry(const std::size_t alloc_size,
      const std::string & tag);
};

}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_USE_GPU_MEMORY_PROFILER
#endif  // MXNET_PROFILER_GPU_MEMORY_PROFILER_H_
