//
// Created by coolivie on 11/25/17.
//

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
 *  Copyright (c) 2017 by Contributors
 * \file c_api_profile.cc
 * \brief C API of mxnet profiler and support functions
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_group.h>
#include <mxnet/kvstore.h>
#include <stack>
#include "./c_api_common.h"
#include "../profiler/storage_profiler.h"
#include "../profiler/profiler.h"

namespace mxnet {

static profiler::ProfileDomain api_domain("MXNET_C_API");
static profiler::ProfileCounter api_call_counter("MXNet C API Calls", &api_domain);
static profiler::ProfileCounter api_concurrency_counter("MXNet C API Concurrency",
                                                        &api_domain);

/*! \brief Per-API-call timing data */
struct APICallTimingData {
  const char *name_;
  profiler::ProfileTask *task_;
};

/*!
 * \brief Per-thread profiling data
 */
class ProfilingThreadData {
 public:
  /*!
   * \brief Constructor, nothrow
   */
  inline ProfilingThreadData() = default;

  /*!
   * \brief Retreive ProfileTask object of the given name, or create if it doesn't exist
   * \param name Name of the task
   * \param domain Domain of the task
   * \return Pointer to the stored or created ProfileTask object
   */
  profiler::ProfileTask *profile_task(const char *name, profiler::ProfileDomain *domain) {
    // Per-thread so no lock necessary
    auto iter = tasks_.find(name);
    if (iter == tasks_.end()) {
      iter = tasks_.emplace(std::make_pair(
        name, std::make_unique<profiler::ProfileTask>(name, domain))).first;
    }
    return iter->second.get();
  }

  /*! \brief nestable call stack */
  std::stack<APICallTimingData> calls_;
  /*! \brief Whether profiling actions should be ignored/excluded */
  volatile bool ignore_call_ = false;  // same-thread only, so not atomic

 private:
  /*! \brief tasks */
  std::unordered_map<std::string, std::unique_ptr<profiler::ProfileTask>> tasks_;
};

#if DMLC_CXX11_THREAD_LOCAL
static thread_local ProfilingThreadData thread_profiling_data;
#else
static MX_THREAD_LOCAL ProfilingThreadData thread_profiling_data;
#endif

extern void on_enter_api(const char *function) {
  if (profiler::Profiler::Get()->IsProfiling(profiler::Profiler::kAPI)) {
    if (!thread_profiling_data.ignore_call_) {
      ++api_call_counter;
      ++api_concurrency_counter;
      APICallTimingData data = {
        function
        , thread_profiling_data.profile_task(function, &api_domain)
      };
      thread_profiling_data.calls_.push(data);
      data.task_->start();
    }
  }
}
extern void on_exit_api() {
  if (profiler::Profiler::Get()->IsProfiling(profiler::Profiler::kAPI)) {
    if (!thread_profiling_data.ignore_call_) {
      CHECK(!thread_profiling_data.calls_.empty());
      APICallTimingData data = thread_profiling_data.calls_.top();
      data.task_->stop();
      thread_profiling_data.calls_.pop();
      --api_concurrency_counter;
    }
  }
}

/*!
 * \brief Don't profile calls in this scope using RAII
 */
struct IgnoreProfileCallScope {
  IgnoreProfileCallScope()  {
    DCHECK_EQ(thread_profiling_data.ignore_call_, false);
    thread_profiling_data.ignore_call_ = true;
  }
  ~IgnoreProfileCallScope() {
    DCHECK_EQ(thread_profiling_data.ignore_call_, true);
    thread_profiling_data.ignore_call_ = false;
  }
};

}  // namespace mxnet

/*!
 * \brief Simple global profile objects created from Python
 * \note These mutexes will almost never have a collision, so internal futexes will be able
 *       to lock in user mode (good performance)
 *       I would use dmlc::SpinLock, except that I am concerned that if conditions change and
 *       there are frequent collisions (ie multithreaded inference), then the spin locks may
 *       start burning CPU unnoticed
 */
struct PythonProfileObjects {
  // These will almost never collide, so locking will happen in user-space (at least on Linux)
  // since pthreads uses futexes.
  std::mutex cs_domains_;
  std::mutex cs_counters_;
  std::mutex cs_tasks_;
  std::mutex cs_frames_;
  std::mutex cs_events_;
  std::list<std::shared_ptr<profiler::ProfileDomain>> domains_;
  std::unordered_map<profiler::ProfileCounter *, std::shared_ptr<profiler::ProfileCounter>>
    counters_;
  std::unordered_map<profiler::ProfileDuration *, std::shared_ptr<profiler::ProfileDuration>>
    tasks_;
  std::unordered_map<profiler::ProfileDuration *, std::shared_ptr<profiler::ProfileDuration>>
    frames_;
  std::unordered_map<profiler::ProfileDuration *, std::shared_ptr<profiler::ProfileDuration>>
    events_;
};
static PythonProfileObjects python_profile_objects;

enum class ProfileProcess {
  kWorker, kServer
};

enum class PrintFormat {
  table, json
};

struct ProfileConfigParam : public dmlc::Parameter<ProfileConfigParam> {
  bool profile_all;
  bool profile_symbolic;
  bool profile_imperative;
  bool profile_memory;
  bool profile_api;
  std::string filename;
  std::string gpu_memory_profile_filename_prefix;
  bool continuous_dump;
  float dump_period;
  bool aggregate_stats;
  int profile_process;
  DMLC_DECLARE_PARAMETER(ProfileConfigParam) {
    DMLC_DECLARE_FIELD(profile_all).set_default(false)
      .describe("Profile all. Default is False.");
    DMLC_DECLARE_FIELD(profile_symbolic).set_default(true)
      .describe("Profile symbolic operators.  Default is True.");
    DMLC_DECLARE_FIELD(profile_imperative).set_default(true)
      .describe("Profile imperative operators.  Default is True.");
    DMLC_DECLARE_FIELD(profile_memory).set_default(true)
      .describe("Profile memory.  Default is True.");
    DMLC_DECLARE_FIELD(profile_api).set_default(true)
      .describe("Profile C API.  Default is True.");
    DMLC_DECLARE_FIELD(filename).set_default("profile.json")
      .describe("File name to write profiling info.");
#if MXNET_USE_CUDA
    DMLC_DECLARE_FIELD(gpu_memory_profile_filename_prefix).set_default("gpu_memory_profile")
      .describe("File name prefix to write GPU memory profile info.");
#endif  // MXNET_USE_CUDA
    DMLC_DECLARE_FIELD(continuous_dump).set_default(true)
      .describe("Periodically dump (and append) profiling data to file while running. "
                "Default is True.");
    DMLC_DECLARE_FIELD(dump_period).set_default(1.0f)
      .describe("When continuous dump is enabled, the period between subsequent "
                  "profile info dumping.");
    DMLC_DECLARE_FIELD(aggregate_stats).set_default(false)
      .describe("Maintain aggregate stats, required for MXDumpAggregateStats.  Note that "
      "this can have a negative performance impact. Default is False.");
    DMLC_DECLARE_FIELD(profile_process)
      .add_enum("worker", static_cast<int>(ProfileProcess::kWorker))
      .add_enum("server", static_cast<int>(ProfileProcess::kServer))
      .set_default(static_cast<int>(ProfileProcess::kWorker))
      .describe("Specifies which process to profile: "
                "worker: this is default. for single node training it should always be worker."
                "server: for distributed training, this profiles server process");
  }
};

DMLC_REGISTER_PARAMETER(ProfileConfigParam);

struct ProfileMarkerScopeParam : public dmlc::Parameter<ProfileMarkerScopeParam> {
  int scope;
  DMLC_DECLARE_PARAMETER(ProfileMarkerScopeParam) {
    DMLC_DECLARE_FIELD(scope).set_default(profiler::ProfileMarker::kProcess)
      .add_enum("global", profiler::ProfileMarker::kGlobal)
      .add_enum("process", profiler::ProfileMarker::kProcess)
      .add_enum("thread", profiler::ProfileMarker::kThread)
      .add_enum("task", profiler::ProfileMarker::kTask)
      .add_enum("marker", profiler::ProfileMarker::kMarker)
      .describe("Profile Instant-Marker scope.");
  }
};

DMLC_REGISTER_PARAMETER(ProfileMarkerScopeParam);

int MXSetProcessProfilerConfig(int num_params, const char* const* keys, const char* const* vals,
                               KVStoreHandle kvstoreHandle) {
    mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    std::vector<std::pair<std::string, std::string>> kwargs;
    kwargs.reserve(num_params);
    for (int i = 0; i < num_params; ++i) {
      CHECK_NOTNULL(keys[i]);
      CHECK_NOTNULL(vals[i]);
      kwargs.emplace_back(std::make_pair(keys[i], vals[i]));
    }
    ProfileConfigParam param;
    param.Init(kwargs);
    if (static_cast<ProfileProcess>(param.profile_process) == ProfileProcess::kServer) {
      std::ostringstream os;
      for (int i = 0; i < num_params; ++i) {
        // this will be sent to the server now, those configs shouldn't have profile server again
        if (strcmp(keys[i], "profile_process") == 0) continue;
        os << keys[i] << ":" << vals[i];
        if (i != num_params - 1) os << ",";
      }
      CHECK(kvstoreHandle) << "KVStoreHandle passed to profiler is null";
      static_cast<KVStore*>(kvstoreHandle)->SetServerProfilerCommand(
      mxnet::KVStoreServerProfilerCommand::kSetConfig, os.str());
    } else {
      int mode = 0;
      if (param.profile_api || param.profile_all)        { mode |= profiler::Profiler::kAPI; }
      if (param.profile_symbolic || param.profile_all)   { mode |= profiler::Profiler::kSymbolic; }
      if (param.profile_imperative ||
          param.profile_all) { mode |= profiler::Profiler::kImperative; }
      if (param.profile_memory || param.profile_all)     { mode |= profiler::Profiler::kMemory; }
      profiler::Profiler::Get()->SetConfig(profiler::Profiler::ProfilerMode(mode),
                                           std::string(param.filename),
                                           param.continuous_dump,
                                           param.dump_period,
                                           param.aggregate_stats);
#if MXNET_USE_CUDA
      profiler::GpuDeviceStorageProfiler::Get()->SetConfig(
          param.gpu_memory_profile_filename_prefix);
#endif  // MXNET_USE_CUDA
    }
  API_END();
}

int MXSetProfilerConfig(int num_params, const char* const* keys, const char* const* vals) {
  return MXSetProcessProfilerConfig(num_params, keys, vals, nullptr);
}

int MXAggregateProfileStatsPrint(const char **out_str, int reset, int format, int sort_by,
                                 int ascending) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
    CHECK_NOTNULL(out_str);
    profiler::Profiler *profiler = profiler::Profiler::Get();
    if (profiler->IsEnableOutput()) {
      // Register stats up until now
      profiler->DumpProfile(false);
    }
    std::shared_ptr<profiler::AggregateStats> stats = profiler->GetAggregateStats();
    std::ostringstream os;
    if (stats) {
      if (static_cast<PrintFormat>(format) == PrintFormat::table)
        stats->DumpTable(os, sort_by, ascending);
      else if (static_cast<PrintFormat>(format) == PrintFormat::json)
        stats->DumpJson(os, sort_by, ascending);
      else
        LOG(FATAL) << "Invalid value for parameter format";
    }
    if (reset != 0)
      stats->clear();
    ret->ret_str = os.str();
    *out_str = (ret->ret_str).c_str();
  API_END();
}

int MXDumpProfile(int finished) {
  return MXDumpProcessProfile(finished, static_cast<int>(ProfileProcess::kWorker), nullptr);
}

int MXDumpProcessProfile(int finished, int profile_process, KVStoreHandle kvStoreHandle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
  if (static_cast<ProfileProcess>(profile_process) == ProfileProcess::kServer) {
    CHECK(kvStoreHandle) << "Kvstore Handle passed to profiler is null";
    static_cast<KVStore*>(kvStoreHandle)->SetServerProfilerCommand(
      mxnet::KVStoreServerProfilerCommand::kDump,
      std::to_string(finished));
  } else {
    profiler::Profiler *profiler = profiler::Profiler::Get();
    CHECK(profiler->IsEnableOutput())
      << "Profiler hasn't been run. Config and start profiler first";
    profiler->DumpProfile(finished != 0);
#if MXNET_USE_CUDA
    profiler::GpuDeviceStorageProfiler::Get()->DumpProfile();
#endif  // MXNET_USE_CUDA
  }
  API_END()
}

int MXSetProfilerState(int state) {
  return MXSetProcessProfilerState(state, static_cast<int>(ProfileProcess::kWorker), nullptr);
}

int MXSetProfilerScope(const char* const scope) {
  API_BEGIN();
  profiler::ProfilerScope::Get()->SetCurrentProfilerScope(scope);
  API_END();
}

int MXSetProcessProfilerState(int state, int profile_process, KVStoreHandle kvStoreHandle) {
  mxnet::IgnoreProfileCallScope ignore;
  // state, kNotRunning: 0, kRunning: 1
  API_BEGIN();
  if (static_cast<ProfileProcess>(profile_process) == ProfileProcess::kServer) {
    CHECK(kvStoreHandle) << "Kvstore Handle passed to profiler is null";
    static_cast<KVStore*>(kvStoreHandle)->SetServerProfilerCommand(
      mxnet::KVStoreServerProfilerCommand::kState,
      std::to_string(state));
  } else {
    switch (state) {
      case profiler::Profiler::kNotRunning:
        profiler::vtune::vtune_pause();
        break;
      case profiler::Profiler::kRunning:
        profiler::vtune::vtune_resume();
        break;
    }
    profiler::Profiler::Get()->SetState(profiler::Profiler::ProfilerState(state));
  }
  API_END();
}

int MXProfileCreateDomain(const char *domain, ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    auto dom = std::make_shared<profiler::ProfileDomain>(domain);
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_domains_);
      python_profile_objects.domains_.push_back(dom);
    }
    *out = dom.get();
  API_END();
}

int MXProfileCreateTask(ProfileHandle domain,
                        const char *task_name,
                        ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    auto ctr =
      std::make_shared<profiler::ProfileTask>(task_name,
                                                static_cast<profiler::ProfileDomain *>(domain));
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_tasks_);
      python_profile_objects.tasks_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
  API_END();
}

int MXProfileCreateFrame(ProfileHandle domain,
                         const char *frame_name,
                         ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    auto ctr =
      std::make_shared<profiler::ProfileFrame>(frame_name,
                                              static_cast<profiler::ProfileDomain *>(domain));
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_frames_);
      python_profile_objects.frames_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
  API_END();
}

int MXProfileCreateEvent(const char *event_name, ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    auto ctr =
      std::make_shared<profiler::ProfileEvent>(event_name);
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_events_);
      python_profile_objects.events_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
  API_END();
}

int MXProfileDestroyHandle(ProfileHandle object_handle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    CHECK_NE(object_handle, static_cast<ProfileHandle>(nullptr))
      << "Invalid NULL handle passed to MXProfileDestroyHandle";
    std::shared_ptr<profiler::ProfileObject> shared_object_ptr(nullptr);
    {
      auto object = static_cast<profiler::ProfileObject *>(object_handle);
      switch (object->type()) {
        case profiler::kTask: {
          auto p = static_cast<profiler::ProfileDuration *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_tasks_);
          auto iter = python_profile_objects.tasks_.find(p);
          if (iter != python_profile_objects.tasks_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.tasks_.erase(iter);
          }
          break;
        }
        case profiler::kEvent: {
          auto p = static_cast<profiler::ProfileDuration *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_events_);
          auto iter = python_profile_objects.events_.find(p);
          if (iter != python_profile_objects.events_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.events_.erase(iter);
          }
          break;
        }
        case profiler::kFrame: {
          auto p = static_cast<profiler::ProfileDuration *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_frames_);
          auto iter = python_profile_objects.frames_.find(p);
          if (iter != python_profile_objects.frames_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.frames_.erase(iter);
          }
          break;
        }
        case profiler::kCounter: {
          auto p = static_cast<profiler::ProfileCounter *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_counters_);
          auto iter = python_profile_objects.counters_.find(p);
          if (iter != python_profile_objects.counters_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.counters_.erase(iter);
          }
          break;
        }
        case profiler::kDomain:
          // Not destroyed
          break;
      }
    }
    shared_object_ptr.reset();  // Destroy out of lock scope
  API_END();
}

int MXProfileDurationStart(ProfileHandle duration_handle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    CHECK_NOTNULL(duration_handle);
    static_cast<profiler::ProfileDuration *>(duration_handle)->start();
  API_END();
}

int MXProfileDurationStop(ProfileHandle duration_handle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    CHECK_NOTNULL(duration_handle);
    static_cast<profiler::ProfileDuration *>(duration_handle)->stop();
  API_END();
}

int MXProfilePause(int paused) {
  return MXProcessProfilePause(paused, static_cast<int>(ProfileProcess::kWorker), nullptr);
}

int MXProcessProfilePause(int paused, int profile_process, KVStoreHandle kvStoreHandle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
  if (static_cast<ProfileProcess>(profile_process) == ProfileProcess::kServer) {
    CHECK(kvStoreHandle) << "Kvstore Handle passed to profiler is null";
    static_cast<KVStore*>(kvStoreHandle)->SetServerProfilerCommand(
      mxnet::KVStoreServerProfilerCommand::kPause,
      std::to_string(paused));
  } else {
    if (paused) {
      profiler::vtune::vtune_pause();
      profiler::Profiler::Get()->set_paused(true);
    } else {
      profiler::Profiler::Get()->set_paused(false);
      profiler::vtune::vtune_resume();
    }
  }
  API_END();
}

int MXProfileCreateCounter(ProfileHandle domain,
                           const char *counter_name,
                           ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    auto ctr =
      std::make_shared<profiler::ProfileCounter>(counter_name,
                                                static_cast<profiler::ProfileDomain *>(domain));
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_counters_);
      python_profile_objects.counters_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
  API_END();
}

int MXProfileSetCounter(ProfileHandle counter_handle, uint64_t value) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    static_cast<profiler::ProfileCounter *>(counter_handle)->operator=(value);
  API_END();
}

int MXProfileAdjustCounter(ProfileHandle counter_handle, int64_t by_value) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    static_cast<profiler::ProfileCounter *>(counter_handle)->operator+=(by_value);
  API_END();
}

int MXProfileSetMarker(ProfileHandle domain,
                       const char *instant_marker_name,
                       const char *scope) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
    ProfileMarkerScopeParam param;
    std::vector<std::pair<std::string, std::string>> kwargs = {{ "scope", scope }};
    param.Init(kwargs);
    profiler::ProfileMarker marker(instant_marker_name,
                                         static_cast<profiler::ProfileDomain *>(domain),
                                         static_cast<profiler::ProfileMarker::MarkerScope>(
                                           param.scope));
    marker.mark();
  API_END();
}
