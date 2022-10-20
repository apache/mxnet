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
#include "./storage_profiler.h"

#if MXNET_USE_NVML
#include <nvml.h>
#endif  // MXNET_USE_NVML
#include <fstream>
#include <map>
#include <regex>
#include <unordered_map>
#include <vector>
#include <type_traits>
#include "./profiler.h"
#include "../common/utils.h"
#include "../common/cuda/utils.h"

namespace mxnet {
namespace profiler {

#if MXNET_USE_CUDA

GpuDeviceStorageProfiler* GpuDeviceStorageProfiler::Get() {
  static std::mutex mtx;
  static std::shared_ptr<GpuDeviceStorageProfiler> gpu_dev_storage_profiler = nullptr;
  std::unique_lock<std::mutex> lk(mtx);
  if (!gpu_dev_storage_profiler) {
    gpu_dev_storage_profiler = std::make_shared<GpuDeviceStorageProfiler>();
  }
  return gpu_dev_storage_profiler.get();
}

#if MXNET_USE_NVML
// Deduce the possibly versioned variant of nvmlProcessInfo_t* expected
// as the 3rd arg of nvmlDeviceGetComputeRunningProcesses().
template <typename F>
struct GetArgType;
template <typename R, typename T1, typename T2, typename T3>
struct GetArgType<R (*)(T1, T2, T3)> {
  typedef T3 arg3_t;
};
using NvmlProcessInfoPtr = GetArgType<decltype(&nvmlDeviceGetComputeRunningProcesses)>::arg3_t;
using NvmlProcessInfo    = std::remove_pointer_t<NvmlProcessInfoPtr>;
#endif

void GpuDeviceStorageProfiler::DumpProfile() const {
  size_t current_pid = common::current_process_id();
  std::ofstream fout((filename_prefix_ + "-pid_" + std::to_string(current_pid) + ".csv").c_str());
  if (!fout.is_open()) {
    return;
  }
  struct AllocEntryDumpFmt {
    size_t requested_size;
    int dev_id;
    size_t actual_size;
    bool reuse;
  };
  // order the GPU memory allocation entries by their attribute name
  std::multimap<std::string, AllocEntryDumpFmt> gpu_mem_ordered_alloc_entries;
  // map the GPU device ID to the total amount of allocations
  std::unordered_map<int, size_t> gpu_dev_id_total_alloc_map;
  std::regex gluon_param_regex(
      "([0-9a-fA-F]{8})_([0-9a-fA-F]{4})_"
      "([0-9a-fA-F]{4})_([0-9a-fA-F]{4})_"
      "([0-9a-fA-F]{12})_([^ ]*)");

  for (const std::pair<void* const, AllocEntry>& alloc_entry : gpu_mem_alloc_entries_) {
    std::string alloc_entry_name =
        std::regex_replace(alloc_entry.second.name, gluon_param_regex, "$6");
    if (alloc_entry_name == "") {
      // If the entry name becomes none after the regex replacement, we revert
      // back to the original.
      alloc_entry_name = alloc_entry.second.name;
    }
    gpu_mem_ordered_alloc_entries.emplace(alloc_entry.second.profiler_scope + alloc_entry_name,
                                          AllocEntryDumpFmt{alloc_entry.second.requested_size,
                                                            alloc_entry.second.dev_id,
                                                            alloc_entry.second.actual_size,
                                                            alloc_entry.second.reuse});
    gpu_dev_id_total_alloc_map[alloc_entry.second.dev_id] = 0;
  }
  fout << "\"Attribute Name\",\"Requested Size\","
          "\"Device\",\"Actual Size\",\"Reuse?\""
       << std::endl;
  for (const std::pair<const std::string, AllocEntryDumpFmt>& alloc_entry :
       gpu_mem_ordered_alloc_entries) {
    fout << "\"" << alloc_entry.first << "\","
         << "\"" << alloc_entry.second.requested_size << "\","
         << "\"" << alloc_entry.second.dev_id << "\","
         << "\"" << alloc_entry.second.actual_size << "\","
         << "\"" << alloc_entry.second.reuse << "\"" << std::endl;
    gpu_dev_id_total_alloc_map[alloc_entry.second.dev_id] += alloc_entry.second.actual_size;
  }
#if MXNET_USE_NVML
  // If NVML has been enabled, add amend term to the GPU memory profile.
  nvmlDevice_t nvml_device;

  NVML_CALL(nvmlInit());
  for (std::pair<const int, size_t>& dev_id_total_alloc_pair : gpu_dev_id_total_alloc_map) {
    unsigned info_count = 0;
    std::vector<NvmlProcessInfo> infos(info_count);

    NVML_CALL(nvmlDeviceGetHandleByIndex(dev_id_total_alloc_pair.first, &nvml_device));
    // The first call to `nvmlDeviceGetComputeRunningProcesses` is to set the
    // size of info. Since `NVML_ERROR_INSUFFICIENT_SIZE` will always be
    // returned, we do not wrap the function call with `NVML_CALL`.
    nvmlDeviceGetComputeRunningProcesses(nvml_device, &info_count, infos.data());
    infos.resize(info_count);
    NVML_CALL(nvmlDeviceGetComputeRunningProcesses(nvml_device, &info_count, infos.data()));

    bool amend_made = false;

    for (unsigned i = 0; i < info_count; ++i) {
      if (current_pid == infos[i].pid) {
        amend_made = true;
        fout << "\""
             << "nvml_amend"
             << "\","
             << "\"" << infos[i].usedGpuMemory - dev_id_total_alloc_pair.second << "\","
             << "\"" << dev_id_total_alloc_pair.first << "\","
             << "\"" << infos[i].usedGpuMemory - dev_id_total_alloc_pair.second << "\","
             << "\"0\"" << std::endl;
        break;
      }
    }
    if (!amend_made) {
      LOG(INFO) << "NVML is unable to make amendment to the GPU memory profile "
                   "since it is unable to locate the current process ID. "
                   "Are you working in Docker without setting --pid=host?";
    }
  }     // for (dev_id_total_alloc_pair : gpu_dev_id_total_alloc_map)
#endif  // MXNET_USE_NVML
}

#endif  // MXNET_USE_CUDA

}  // namespace profiler
}  // namespace mxnet
