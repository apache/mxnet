/*!
 * Copyright (c) 2015 by Contributors
 * \file profiler.cc
 * \brief implements profiler
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <set>
#include <map>
#include <mutex>
#include <chrono>
#include <iostream>
#include <fstream>
#include "./profiler.h"

namespace mxnet {
namespace engine {

// TODO(ziheng) more lock free

Profiler* Profiler::instance_ = nullptr;
std::mutex Profiler::m_;

Profiler::Profiler()
    : state_(kNotRunning), enable_output_(false), mode_(kOnlySymbolic), filename_("profile.json") {
  this->init_time_ = NowInUsec();

  // TODO(ziheng) get device number during execution
  int kMaxNumCpus = 60, kMaxNumGpus = 16;
  this->cpu_num_ = kMaxNumCpus, this->gpu_num_ = kMaxNumGpus;

  this->profile_stat = new DevStat[cpu_num_ + gpu_num_ + 1];
  for (unsigned int i = 0; i < cpu_num_; ++i) {
    profile_stat[i].dev_name = "cpu/" + std::to_string(i);
  }
  for (unsigned int i = 0; i < gpu_num_; ++i) {
    profile_stat[cpu_num_ + i].dev_name = "gpu/" + std::to_string(i);
  }
  profile_stat[cpu_num_ + gpu_num_].dev_name = "cpu pinned/";
}

Profiler* Profiler::Get() {
  std::lock_guard<std::mutex> lock{Profiler::m_};
  if (instance_ == nullptr) {
    instance_ = new Profiler;
  }
  return instance_;
}

void Profiler::SetState(ProfilerState state) {
  std::lock_guard<std::mutex> lock{Profiler::m_};
  this->state_ = state;
  // once running, output will be enabled.
  if (state == kRunning)
      this->enable_output_ = true;
}

void Profiler::SetConfig(ProfilerMode mode, std::string output_filename) {
  std::lock_guard<std::mutex> lock{Profiler::m_};
  this->mode_ = mode;
  this->filename_ = output_filename;
}

OprExecStat *Profiler::AddOprStat(int dev_type, int dev_id) {
  std::lock_guard<std::mutex> lock{Profiler::m_};

  OprExecStat* opr_stat = new OprExecStat;
  opr_stat->dev_type = dev_type;
  opr_stat->dev_id   = dev_id;

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
  dev_stat.opr_exec_stats.push_back(opr_stat);

  return opr_stat;
}

void Profiler::EmitPid(std::ostream *os, const std::string& name, int pid) {
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
                       uint64_t ts, int pid, int tid) {
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
  std::lock_guard<std::mutex> lock{Profiler::m_};
  std::ofstream file;
  file.open(filename_);

  file << "{" << std::endl;
  file << "    \"traceEvents\": [" << std::endl;

  int dev_num = cpu_num_ + gpu_num_ + 1;

  for (int i = 0; i < dev_num; ++i) {
    const DevStat &d = profile_stat[i];
    this->EmitPid(&file, d.dev_name, i);
    file << ",\n";
  }

  bool first_flag = true;
  for (int i = 0; i < dev_num; ++i) {
    const DevStat &d = profile_stat[i];
    int opr_num = d.opr_exec_stats.size();

    for (int j = 0; j < opr_num; ++j) {
      const OprExecStat* opr_stat = d.opr_exec_stats[j];

      int pid = i;
      int tid = opr_stat->thread_id;

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
            opr_stat->opr_end_rel_micros,   pid, tid);
    }
  }

  file << "\n" << std::endl;
  file << "    ]," << std::endl;
  file << "    \"displayTimeUnit\": \"ms\"" << std::endl;
  file << "}" << std::endl;
}


inline uint64_t NowInUsec() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
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
