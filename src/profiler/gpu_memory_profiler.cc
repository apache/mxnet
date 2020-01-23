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
#include "./gpu_memory_profiler.h"
#include <dmlc/parameter.h>
#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#endif
#include <cstdint>
#include <ctime>
#include <string>

namespace mxnet {
namespace profiler {

/// \brief Return the path to the temporary directory.
///        The implementation is taken from `boost::temp_directory_path`
static std::string temp_dir_path() {
#ifdef _WIN32
#define TEMP_DIR_MAX_LEN 1024
  TCHAR temp_dir_c_str[TEMP_DIR_MAX_LEN];
  GetTempPath(TEMP_DIR_MAX_LEN,
      temp_dir_c_str);
  return std::string(temp_dir_c_str);
#undef MAX_PATH
#else  // !_WIN32
  const char* temp_dir_c_str = nullptr;

  (temp_dir_c_str = std::getenv("TMPDIR")) ||
  (temp_dir_c_str = std::getenv("TMP")) ||
  (temp_dir_c_str = std::getenv("TEMP")) ||
  (temp_dir_c_str = std::getenv("TEMPDIR"));
#ifdef __ANDROID__
  const char* default_temp_dir_c_str = "/data/local/tmp";
#else
  const char* default_temp_dir_c_str = "/tmp";
#endif
  std::string temp_dir = std::string(temp_dir_c_str == nullptr ?
      default_temp_dir_c_str : temp_dir_c_str);
  return temp_dir;
#endif  // _WIN32
}

/// \brief Convert a `Storage::DataStruct` to string.
static std::string data_struct_to_str(
    const Storage::DataStruct data_struct) {
  switch (data_struct) {
    case Storage::DataStruct::kDataEntry:      return "data entry";
    case Storage::DataStruct::kTempSpace:      return "temp space";
    case Storage::DataStruct::kParameter:      return "param";
    case Storage::DataStruct::kParameterGrad:  return "param grad";
    case Storage::DataStruct::kOptimizerState: return "optimizer state";
    case Storage::DataStruct::kAuxState:       return "aux state";
    case Storage::DataStruct::kEphemeral:      return "ephemeral";
    default: return "unknown";
  }
}

GpuMemoryProfiler* GpuMemoryProfiler::Get() {
  static GpuMemoryProfiler s_gpu_memory_profiler;
  return &s_gpu_memory_profiler;
}

GpuMemoryProfiler::GpuMemoryProfiler() {
  enabled_ = dmlc::GetEnv("MXNET_USE_PROFILER", false);

  if (enabled_) {
    std::string temp_dir = temp_dir_path(),
                alloc_csv_fname = temp_dir + "mxnet_gpu_memory_profiler_alloc.csv";

    alloc_csv_fout_.open(alloc_csv_fname.c_str());
    CHECK_EQ(alloc_csv_fout_.is_open(), true);
    // add the CSV header
    alloc_csv_fout_ << "\"Time Stamp\",\"Scope\",\"Attribute Name\","
                       "\"Data Struct\","
                       "\"Requested\",\"Actual\",\"Reuse?\""
                    << std::endl;
  }  // enabled_
}

char GpuMemoryProfiler::current_profiler_scope_[PROFILER_SCOPE_MAX_LEN] =
    MXNET_STORAGE_DEFAULT_PROFILER_SCOPE_CSTR;

void GpuMemoryProfiler::SetCurrentScope(const std::string& scope) {
  strncpy(current_profiler_scope_, scope.c_str(),
          PROFILER_SCOPE_MAX_LEN);
}

std::string GpuMemoryProfiler::GetCurrentScope() {
  return std::string(current_profiler_scope_);
}

void GpuMemoryProfiler::addEntry(const Storage::Handle& handle,
    const size_t actual_alloc_size, const bool reuse) {
  // directly skip the recording if the memory profiler is not enabled or
  // the storage handle has been marked as 'ephemeral'
  if (!enabled_ ||
      handle.data_struct == Storage::DataStruct::kEphemeral) {
    return;
  }
  alloc_csv_fout_ << static_cast<uint64_t>(time(nullptr)) << ","
                  << "\"" << handle.profiler_scope << "\","
                  << "\"" << handle.name  << "\","
                  << "\"" << data_struct_to_str(handle.data_struct) << "\","
                  << handle.size << ","
                  << actual_alloc_size << ","
                  << reuse << std::endl;
}

}  // namespace profiler
}  // namespace mxnet
