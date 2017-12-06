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
 * Copyright (c) 2015 by Contributors
 * \file profiler.cc
 * \brief implements profiler
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/omp.h>
#include <mxnet/base.h>
#include <fstream>
#include <thread>
#include "./profiler.h"

#if MXNET_USE_CUDA
#include "../common/cuda_utils.h"
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1800
#include <Windows.h>
#endif

namespace mxnet {
namespace engine {

Profiler::Profiler()
  : state_(kNotRunning), enable_output_(false), filename_("profile.json") {
  this->init_time_ = NowInUsec();

  this->cpu_num_ = std::thread::hardware_concurrency();
#if MXNET_USE_CUDA
  int kMaxNumGpus = 32;
  this->gpu_num_ = kMaxNumGpus;
#else
  this->gpu_num_ = 0;
#endif

  this->profile_stat = new DevStat[cpu_num_ + gpu_num_ + 1];
  for (unsigned int i = 0; i < cpu_num_; ++i) {
    profile_stat[i].dev_name_ = "cpu/" + std::to_string(i);
  }
  for (unsigned int i = 0; i < gpu_num_; ++i) {
    profile_stat[cpu_num_ + i].dev_name_ = "gpu/" + std::to_string(i);
  }
  profile_stat[cpu_num_ + gpu_num_].dev_name_ = "cpu pinned/";

  mode_ = (ProfilerMode)dmlc::GetEnv("MXNET_PROFILER_MODE", static_cast<int>(kOnlySymbolic));
  if (dmlc::GetEnv("MXNET_PROFILER_AUTOSTART", 0)) {
    this->state_ = ProfilerState::kRunning;
    this->enable_output_ = true;
  }
}

Profiler* Profiler::Get() {
#if MXNET_USE_PROFILER
  static Profiler inst;
  return &inst;
#else
  return nullptr;
#endif
}

void Profiler::SetState(ProfilerState state) {
  std::lock_guard<std::mutex> lock{this->m_};
  this->state_ = state;
  // once running, output will be enabled.
  if (state == kRunning)
      this->enable_output_ = true;
}

void Profiler::SetConfig(ProfilerMode mode, std::string output_filename) {
  std::lock_guard<std::mutex> lock{this->m_};
  this->mode_ = mode;
  this->filename_ = output_filename;
}

OprExecStat *Profiler::AddOprStat(int dev_type, uint32_t dev_id) {
  std::unique_ptr<OprExecStat> opr_stat(new OprExecStat);
  opr_stat->dev_type = dev_type;
  opr_stat->dev_id   = dev_id;
  opr_stat->opr_name[sizeof(opr_stat->opr_name)-1] = '\0';

  int idx;
  switch (dev_type) {
    case Context::kCPU:
      idx = dev_id;
      break;
    case Context::kGPU:
      idx = cpu_num_ + dev_id;
      break;
    case Context::kCPUPinned:
      idx = cpu_num_ + gpu_num_;
      break;
    default:
      LOG(FATAL) << "Unknown dev_type: " << dev_type;
      return NULL;
  }

  DevStat& dev_stat = profile_stat[idx];
  dev_stat.opr_exec_stats_->enqueue(opr_stat.get());
  return opr_stat.release();
}

void Profiler::EmitPid(std::ostream *os, const std::string& name, uint32_t pid) {
  (*os) << "        {\n"
        << "            \"ph\": \"M\",\n"
        << "            \"args\": {\n"
        << "                \"name\": \"" << name << "\"\n"
        << "            },\n"
        << "            \"pid\": " << pid << ",\n"
        << "            \"name\": \"process_name\"\n"
        << "        }";
}

void Profiler::EmitEvent(std::ostream *os, const std::string& name,
                       const std::string& category, const std::string& ph,
                       uint64_t ts, uint32_t pid, uint32_t tid) {
  (*os) << "        {\n"
        << "            \"name\": \""  << name << "\",\n"
        << "            \"cat\": " << "\"" << category << "\",\n"
        << "            \"ph\": \""<< ph << "\",\n"
        << "            \"ts\": "  << ts << ",\n"
        << "            \"pid\": " << pid << ",\n"
        << "            \"tid\": " << tid << "\n"
        << "        }";
}


void Profiler::DumpProfile() {
  SetState(kNotRunning);

  std::lock_guard<std::mutex> lock{this->m_};
  std::ofstream file;
  file.open(filename_);

  file << "{" << std::endl;
  file << "    \"traceEvents\": [" << std::endl;

  uint32_t dev_num = cpu_num_ + gpu_num_ + 1;

  for (uint32_t i = 0; i < dev_num; ++i) {
    const DevStat &d = profile_stat[i];
    this->EmitPid(&file, d.dev_name_, i);
    file << ",\n";
  }

  bool first_flag = true;
  for (uint32_t i = 0; i < dev_num; ++i) {
    DevStat &d = profile_stat[i];
    OprExecStat *_opr_stat;
    while (d.opr_exec_stats_->try_dequeue(_opr_stat)) {
      CHECK_NOTNULL(_opr_stat);
      std::unique_ptr<OprExecStat> opr_stat(_opr_stat);  // manage lifecycle
      uint32_t pid = i;
      uint32_t tid = opr_stat->thread_id;

      if (first_flag) {
        first_flag = false;
      } else {
        file << ",";
      }
      file << std::endl;
      this->EmitEvent(&file, opr_stat->opr_name, "category", "B",
                      opr_stat->opr_start_rel_micros, pid, tid);
      file << ",\n";
      this->EmitEvent(&file, opr_stat->opr_name, "category", "E",
                      opr_stat->opr_end_rel_micros, pid, tid);
    }
  }

  file << "\n" << std::endl;
  file << "    ]," << std::endl;
  file << "    \"displayTimeUnit\": \"ms\"" << std::endl;
  file << "}" << std::endl;

  enable_output_ = false;
}


inline uint64_t NowInUsec() {
#if defined(_MSC_VER) && _MSC_VER <= 1800
  LARGE_INTEGER frequency, counter;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&counter);
  return counter.QuadPart * 1000000 / frequency.QuadPart;
#else
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
}

void SetOprStart(OprExecStat* opr_stat) {
  if (!opr_stat) {
    LOG(WARNING) << "SetOpStart: nullptr";
    return;
  }
  opr_stat->opr_start_rel_micros = NowInUsec() - Profiler::Get()->GetInitTime();
}

void SetOprEnd(OprExecStat* opr_stat) {
  if (!opr_stat) {
    LOG(WARNING) << "SetOpEnd: nullptr";
    return;
  }
  opr_stat->opr_end_rel_micros   = NowInUsec() - Profiler::Get()->GetInitTime();
}

}  // namespace engine
}  // namespace mxnet
