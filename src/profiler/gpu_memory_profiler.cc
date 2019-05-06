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
 * \file gpu_memory_profiler.cc
 * \author Bojian Zheng and Abhishek Tiwari
 * \brief implementation of the `GpuMemoryProfiler` class
 */

#include "./gpu_memory_profiler.h"

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <cstring>  // strcmp

#if MXNET_USE_GPU_MEMORY_PROFILER

namespace mxnet {
namespace profiler {

GpuMemoryProfiler *
GpuMemoryProfiler::Get() {
  static GpuMemoryProfiler s_gpu_memory_profiler;
  return &s_gpu_memory_profiler;
}

GpuMemoryProfiler::GpuMemoryProfiler() {
  bool use_gpu_memory_profiler = dmlc::GetEnv("MXNET_USE_GPU_MEMORY_PROFILER", false);
  std::string csv_fname = dmlc::GetEnv("MXNET_GPU_MEMORY_PROFILER_CSV_FNAME",
      std::string("mxnet_gpu_memory_profiler.csv"));
  std::string log_fname = dmlc::GetEnv("MXNET_GPU_MEMORY_PROFILER_LOG_FNAME",
      std::string("mxnet_gpu_memory_profiler.log"));

  if (!use_gpu_memory_profiler) {
    LOG(INFO) << "Set MXNET_USE_GPU_MEMORY_PROFILER=1 "
              << "to enable the GPU memory profiler.";
    _enabled = false;
  } else {
    LOG(INFO) << "MXNet has the GPU memory profiler enabled.";
    _enabled = true;
    _csv_fout.open(csv_fname.c_str());
    _log_fout.open(log_fname.c_str());
    CHECK_EQ(_csv_fout.is_open(), true);
    CHECK_EQ(_log_fout.is_open(), true);
  }
}

void
GpuMemoryProfiler::addEntry(
    const std::size_t alloc_size,
    const std::string & tag) {
  if (!_enabled || (tag == "skip me")) {
    return;
  }
#define MiB (1024.0 * 1024.0)
  _csv_fout << tag << "," << alloc_size / MiB << std::endl;
  // If "unknown" is part of the `tag`, log the entry together with the callstack.
  if (tag.find("unknown") != std::string::npos) {
    std::string stack_trace =
#if DMLC_LOG_STACK_TRACE
        dmlc::StackTrace(2ul);
#else  // !DMLC_LOG_STACK_TRACE
        "";
#endif  // DMLC_LOG_STACK_TRACE
    _log_fout << "[mxnet:profiler:gpu_memory_profiler:W] "
              << "Allocating " << alloc_size / MiB
              << " with unknown memory tag " << tag
              << std::endl << stack_trace << std::endl;
  }
#undef MiB
}

}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_USE_GPU_MEMORY_PROFILER
