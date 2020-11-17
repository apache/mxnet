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
#include "../common/cuda/utils.h"
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1800
#include <Windows.h>
#endif

namespace mxnet {
namespace profiler {

ProfileDomain ProfileOperator::domain_("operator");

Profiler::Profiler()
  : state_(kNotRunning)
    , enable_output_(false)
    , filename_("profile.json")
    , profile_dump_count_(0) {
  this->init_time_ = ProfileStat::NowInMicrosec();

  this->cpu_num_ = 0;
#ifdef _OPENMP
  // OMP is going to know better than a generic stl, which might return 0
  this->cpu_num_ = static_cast<unsigned int>(::omp_get_num_procs());
#else
  this->cpu_num_ = std::thread::hardware_concurrency();
#endif
  if (!this->cpu_num_) {
    this->cpu_num_ = 64;
  }
#if MXNET_USE_CUDA
  int kMaxNumGpus = 32;
  this->gpu_num_ = kMaxNumGpus;
#else
  this->gpu_num_ = 0;
#endif

  this->profile_stat = std::unique_ptr<DeviceStats[]>(new DeviceStats[cpu_num_ + gpu_num_ + 2]);
  for (unsigned int i = 0; i < cpu_num_; ++i) {
    this->profile_stat[i].dev_name_ = "cpu/" + std::to_string(i);
  }
  for (unsigned int i = 0; i < gpu_num_; ++i) {
    this->profile_stat[cpu_num_ + i].dev_name_ = "gpu/" + std::to_string(i);
  }
  this->profile_stat[cpu_num_ + gpu_num_].dev_name_ = "cpu pinned/";

  this->profile_stat[cpu_num_ + gpu_num_ + 1].dev_name_ = "cpu shared/";

  this->mode_ = dmlc::GetEnv("MXNET_PROFILER_MODE", this->mode_);
  if (dmlc::GetEnv("MXNET_PROFILER_AUTOSTART", 0)) {
    this->state_ = ProfilerState::kRunning;
    this->enable_output_ = true;
    // Since we want to avoid interfering with pure-VTune analysis runs, for not set,
    // vtune will be recording based upon whether "Start" or "STart Paused" was selected
    vtune::vtune_resume();
  }
}

Profiler::~Profiler() {
  DumpProfile(true);
  if (thread_group_) {
    thread_group_->request_shutdown_all();
    thread_group_->join_all();
    thread_group_.reset();
  }
}

Profiler* Profiler::Get(std::shared_ptr<Profiler> *sp) {
  static std::mutex mtx;
  static std::shared_ptr<Profiler> prof = nullptr;
  if (!prof) {
    std::unique_lock<std::mutex> lk(mtx);
    if (!prof) {
      prof = std::make_shared<Profiler>();
    }
  }
  if (sp) {
    *sp = prof;
  }
  return prof.get();
}

void Profiler::SetState(ProfilerState state) {
  std::lock_guard<std::recursive_mutex> lock{this->m_};
  this->state_ = state;
  // once running, output will be enabled.
  if (state == kRunning) {
    this->enable_output_ = true;
    set_paused(false);
  } else {
    set_paused(true);
  }
}

void Profiler::SetConfig(int mode,
                         std::string output_filename,
                         bool continuous_dump,
                         float dump_period,
                         bool aggregate_stats) {
  CHECK(!continuous_dump || dump_period > 0);
  std::lock_guard<std::recursive_mutex> lock{this->m_};
  this->mode_ = mode;
  this->filename_ = output_filename;
  // Remove the output file to start
  if (!this->filename_.empty()) {
    ::unlink(this->filename_.c_str());
  }
  SetContinuousProfileDump(continuous_dump, dump_period);
  // Adjust whether storing aggregate stats as necessary
  if (aggregate_stats) {
    if (!aggregate_stats_) {
      aggregate_stats_ = std::make_shared<AggregateStats>();
    }
  } else if (aggregate_stats_) {
    aggregate_stats_.reset();
  }
}

/*
 * Docs for tracing format:
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
void Profiler::EmitPid(std::ostream *os, const std::string& name, size_t pid) {
  (*os) << "        {\n"
        << R"(            "ph": ")" << static_cast<char>(ProfileStat::kMetadata) <<  "\",\n"
        << "            \"args\": {\n"
        << R"(                "name": ")" << name << "\"\n"
        << "            },\n"
        << "            \"pid\": " << pid << ",\n"
        << "            \"name\": \"process_name\"\n"
        << "        }";
}

void Profiler::DumpProfile(bool perform_cleanup) {
  std::lock_guard<std::recursive_mutex> lock{this->m_};
  if (!IsEnableOutput()) {
    return;
  }
  if (perform_cleanup) {
    SetContinuousProfileDump(false, 1.0f);
  }
  std::ofstream file;
  const bool first_pass = ++profile_dump_count_ == 1;
  const bool last_pass = perform_cleanup || !continuous_dump_;
  if (!first_pass && continuous_dump_) {
    file.open(filename_, std::ios::app|std::ios::out);
  } else {
    file.open(filename_, std::ios::trunc|std::ios::out);
  }
  if (first_pass || !continuous_dump_) {
    file << "{" << std::endl;
    file << "    \"traceEvents\": [" << std::endl;
  }

  const size_t dev_num = DeviceCount();

  if (first_pass) {
    for (uint32_t pid = 0; pid < dev_num; ++pid) {
       if (pid) {
         file << ",\n";
       }
      const DeviceStats &d = profile_stat[pid];
      this->EmitPid(&file, d.dev_name_, pid);
      process_ids_.emplace(pid);
    }
  }

  // Hold ref in case SetConfig() resets aggregate_stats_
  // If aggregate stats aren't enabled, this won't cause a locked instruction
  std::shared_ptr<AggregateStats> ptr_aggregate_stats = aggregate_stats_.get()
                                                        ? aggregate_stats_ : nullptr;
  for (uint32_t i = 0; i < dev_num; ++i) {
    DeviceStats &d = profile_stat[i];
    ProfileStat *_opr_stat;
    while (d.opr_exec_stats_->try_dequeue(_opr_stat)) {
      CHECK_NOTNULL(_opr_stat);
      std::unique_ptr<ProfileStat> opr_stat(_opr_stat);  // manage lifecycle
      opr_stat->process_id_ = i;  // lie and set process id to be the device number
      file << ",\n" << std::endl;
      opr_stat->EmitEvents(&file);
      ++num_records_emitted_;
      if (ptr_aggregate_stats) {
        ptr_aggregate_stats->OnProfileStat(*_opr_stat);
      }
    }
  }

  // Now do the non-device items
  ProfileStat *_profile_stat;
  while (general_stats_.opr_exec_stats_->try_dequeue(_profile_stat)) {
    CHECK_NOTNULL(_profile_stat);
    file << ",";
    std::unique_ptr<ProfileStat> profile_stat(_profile_stat);  // manage lifecycle
    CHECK_NE(profile_stat->categories_.c_str()[0], '\0') << "Category must be set";
    // Currently, category_to_pid_ is only accessed here, so it is protected by this->m_ above
    auto iter = category_to_pid_.find(profile_stat->categories_.c_str());
    if (iter == category_to_pid_.end()) {
      static std::hash<std::string> hash_fn;
      const size_t this_pid = hash_fn(profile_stat->categories_.c_str());
      iter = category_to_pid_.emplace(std::make_pair(profile_stat->categories_.c_str(),
                                                     this_pid)).first;
      EmitPid(&file, profile_stat->categories_.c_str(), iter->second);
      file << ",\n";
    }
    profile_stat->process_id_ = iter->second;
    file << std::endl;
    profile_stat->EmitEvents(&file);
    ++num_records_emitted_;
    if (ptr_aggregate_stats) {
      ptr_aggregate_stats->OnProfileStat(*profile_stat);
    }
  }

  if (last_pass) {
    file << "\n" << std::endl;
    file << "    ]," << std::endl;
    file << R"(    "displayTimeUnit": "ms")" << std::endl;
    file << "}" << std::endl;
  }
  enable_output_ = continuous_dump_ && !last_pass;  // If we're appending, then continue.
                                                    // Otherwise, profiling stops.
}

static constexpr char TIMER_THREAD_NAME[] = "DumpProfileTimer";

void Profiler::SetContinuousProfileDump(bool continuous_dump, float delay_in_seconds) {
  std::lock_guard<std::recursive_mutex> lock{this->m_};
  if (continuous_dump) {
    this->continuous_dump_ = true;  // Continuous doesn't make sense without append mode
    DumpProfile(false);
    std::shared_ptr<dmlc::ThreadGroup::Thread> old_thread =
      thread_group_->thread_by_name(TIMER_THREAD_NAME);
    if (old_thread && old_thread->is_shutdown_requested()) {
      // This should never happen unless someone is doing something malicious
      // At any rate, wait for its shutdown to complete
      if (old_thread->joinable()) {
        old_thread->join();
      } else {
        do {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (thread_group_->thread_by_name(TIMER_THREAD_NAME));
      }
      old_thread.reset();
    }
    if (!old_thread) {
      dmlc::CreateTimer(
        TIMER_THREAD_NAME,
        std::chrono::milliseconds(static_cast<size_t>(delay_in_seconds * 1000.0f)),
        thread_group_.get(),
        [this]() -> int {
          DumpProfile(false);
          return 0;
        });
    }
  } else {
    std::shared_ptr<dmlc::ThreadGroup::Thread> old_thread =
      thread_group_->thread_by_name(TIMER_THREAD_NAME);
    if (old_thread) {
      // Signal it to finish asynchronously
      old_thread->request_shutdown();
    }
  }
}

ProfilerScope* ProfilerScope::Get() {
  static std::mutex mtx;
  static std::shared_ptr<ProfilerScope> prof_scope = nullptr;
  std::unique_lock<std::mutex> lk(mtx);
  if (!prof_scope) {
    prof_scope = std::make_shared<ProfilerScope>();
  }
  return prof_scope.get();
}

void ProfilerScope::SetCurrentProfilerScope(const std::string& scope) {
  current_profiler_scope_ = scope;
}

std::string ProfilerScope::GetCurrentProfilerScope() const {
  return current_profiler_scope_;
}

}  // namespace profiler
}  // namespace mxnet
