/*!
 * Copyright (c) 2015 by Contributors
 * \file profiler.cc
 * \brief implements profiler
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <set>
#include <map>
#include <mutex>
#include <chrono>
#include <iostream>
#include <fstream>
#include "./profiler.h"

#if defined(_MSC_VER) && _MSC_VER <= 1800
#include <Windows.h>
#endif

namespace mxnet {
namespace engine {
const int INITIAL_SIZE = 1024;

Profiler::Profiler()
  : state_(kNotRunning), enable_output_(false), filename_("profile.json") {
  this->init_time_ = NowInUsec();

  // TODO(ziheng) get device number during execution
  int kMaxNumCpus = 64;
  this->cpu_num_ = kMaxNumCpus;
#if MXNET_USE_CUDA
  int kMaxNumGpus = 32;
  this->gpu_num_ = kMaxNumGpus;
#else
  this->gpu_num_ = 0;
#endif

  this->profile_stat = new DevStat[cpu_num_ + gpu_num_ + 1];
  this->profile_stat->opr_exec_stats.reserve(INITIAL_SIZE);
  for (unsigned int i = 0; i < cpu_num_; ++i) {
    profile_stat[i].dev_name = "cpu/" + std::to_string(i);
  }
  for (unsigned int i = 0; i < gpu_num_; ++i) {
    profile_stat[cpu_num_ + i].dev_name = "gpu/" + std::to_string(i);
  }
  profile_stat[cpu_num_ + gpu_num_].dev_name = "cpu pinned/";

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
  OprExecStat* opr_stat = new OprExecStat;
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
      LOG(FATAL) << "Unkown dev_type";
      return NULL;
  }

  DevStat& dev_stat = profile_stat[idx];
  {
    std::lock_guard<std::mutex> lock{dev_stat.m_};
    dev_stat.opr_exec_stats.push_back(opr_stat);
  }
  return opr_stat;
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
    this->EmitPid(&file, d.dev_name, i);
    file << ",\n";
  }

  bool first_flag = true;
  for (uint32_t i = 0; i < dev_num; ++i) {
    DevStat &d = profile_stat[i];
    std::lock_guard<std::mutex> lock(d.m_);
    uint32_t opr_num = d.opr_exec_stats.size();

    for (uint32_t j = 0; j < opr_num; ++j) {
      const OprExecStat* opr_stat = d.opr_exec_stats[j];

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
