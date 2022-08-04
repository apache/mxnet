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
 * \file cudnn_cxx.cc
 */
#include "cudnn_cxx.h"

#include <mxnet/base.h>
#if MXNET_USE_CUDNN == 1

#include <mxnet/storage.h>
#include <algorithm>
#include <sstream>
#include <utility>

namespace mxnet {
namespace cudnn_cxx {

Descriptor Make(cudnnBackendDescriptorType_t type) {
  cudnnBackendDescriptor_t desc{};
  CUDNN_CALL(cudnnBackendCreateDescriptor(type, &desc));
  return Descriptor(desc);
}

std::vector<cudnnBackendDescriptor_t> MakeRawDescriptors(size_t n,
                                                         cudnnBackendDescriptorType_t type) {
  std::vector<cudnnBackendDescriptor_t> ret(n);
  for (auto& d : ret)
    CUDNN_CALL(cudnnBackendCreateDescriptor(type, &d));
  return ret;
}

void SetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name, const Descriptor& val) {
  auto raw = val.get();
  CUDNN_CALL(cudnnBackendSetAttribute(desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw));
}

void SetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name, const WeakDescriptor& val) {
  auto raw = val.get();
  CUDNN_CALL(cudnnBackendSetAttribute(desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw));
}

void SetAttr(const Descriptor& desc,
             cudnnBackendAttributeName_t name,
             const std::vector<Descriptor>& val) {
  std::vector<cudnnBackendDescriptor_t> raw(val.size());
  std::transform(val.begin(), val.end(), raw.begin(), [](const Descriptor& d) { return d.get(); });
  CUDNN_CALL(cudnnBackendSetAttribute(
      desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, raw.size(), raw.data()));
}

Descriptor GetAttr(const Descriptor& desc,
                   cudnnBackendAttributeName_t name,
                   cudnnBackendDescriptorType_t type) {
  cudnnBackendDescriptor_t ret{};
  CUDNN_CALL(cudnnBackendCreateDescriptor(type, &ret));
  int64_t count = 0;
  CUDNN_CALL(
      cudnnBackendGetAttribute(desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &ret));
  CHECK_EQ(count, 1);
  return Descriptor(ret);
}

std::vector<Descriptor> GetAllAttrs(const Descriptor& desc,
                                    cudnnBackendAttributeName_t name,
                                    cudnnBackendDescriptorType_t type) {
  int64_t count = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(
      desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
  auto raw = MakeRawDescriptors(count, type);
  CUDNN_CALL(cudnnBackendGetAttribute(
      desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, raw.size(), &count, raw.data()));

  CHECK_LE(count, raw.size());
  std::vector<Descriptor> ret(raw.begin(), raw.begin() + count);
  for (size_t i = count; i < raw.size(); ++i)
    CUDNN_CALL(cudnnBackendDestroyDescriptor(raw[i]));
  return ret;
}

std::vector<Descriptor> GetSomeAttrs(size_t max_n,
                                     const Descriptor& desc,
                                     cudnnBackendAttributeName_t name,
                                     cudnnBackendDescriptorType_t type) {
  auto raw      = MakeRawDescriptors(max_n, type);
  int64_t count = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(
      desc.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, raw.size(), &count, raw.data()));
  std::vector<Descriptor> ret(count);
  size_t i = 0;
  for (; i < count; ++i)
    ret[i] = Descriptor(raw[i]);
  for (; i < max_n; ++i)
    CUDNN_CALL(cudnnBackendDestroyDescriptor(raw[i]));
  return ret;
}

std::vector<Descriptor> GetPlans(cudnnBackendHeurMode_t h_mode,
                                 cudnnHandle_t handle,
                                 const Descriptor& op_graph,
                                 size_t workspace_limit,
                                 size_t* max_workspace,
                                 const std::unordered_set<int64_t>& excl_engines,
                                 const std::vector<cudnnBackendNumericalNote_t>& req_numeric,
                                 const std::vector<cudnnBackendNumericalNote_t>& excl_numeric,
#if CUDNN_VERSION >= 8200
                                 const std::vector<cudnnBackendBehaviorNote_t>& req_behavior,
                                 const std::vector<cudnnBackendBehaviorNote_t>& excl_behavior,
#endif  // CUDNN_VERSION >= 8200
                                 bool verbose_filter) {
  auto heur = MakeFinalized(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                            CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                            op_graph,
                            CUDNN_ATTR_ENGINEHEUR_MODE,
                            h_mode);
  auto cfgs = GetAllAttrs(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
  std::vector<Descriptor> plans;
  if (max_workspace)
    *max_workspace = 0;
  for (const auto& cfg : cfgs) {
    auto plan = Make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                     CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                     handle,
                     CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                     cfg);
    auto err  = cudnnBackendFinalize(plan.get());
    if (err == CUDNN_STATUS_NOT_SUPPORTED || err == CUDNN_STATUS_ARCH_MISMATCH)
      continue;
    if (err != CUDNN_STATUS_SUCCESS) {
      LOG(WARNING) << "Unexpected cuDNN status: " << err << ": " << cudnnGetErrorString(err);
      continue;
    }
    auto workspace = GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
    if (workspace_limit < workspace) {
      if (verbose_filter)
        LOG(INFO) << "   Plan " << PlanStr(plan) << " exceeds workspace limit";
      continue;
    }
    auto engine = GetAttr(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    if (excl_engines.count(GetAttr<int64_t>(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX))) {
      if (verbose_filter)
        LOG(INFO) << "   Plan " << PlanStr(plan) << " excluded by engine";
      continue;
    }
    auto numerical = GetSomeAttrs<cudnnBackendNumericalNote_t>(
        CUDNN_NUMERICAL_NOTE_TYPE_COUNT, engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE);
    if (!IsCompatible(numerical, req_numeric, excl_numeric)) {
      if (verbose_filter)
        LOG(INFO) << "   Plan " << PlanStr(plan) << " has incompatible numerics";
      continue;
    }
#if CUDNN_VERSION >= 8200
    auto behavior = GetSomeAttrs<cudnnBackendBehaviorNote_t>(
        CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE);
    if (!IsCompatible(behavior, req_behavior, excl_behavior)) {
      if (verbose_filter)
        LOG(INFO) << "   Plan " << PlanStr(plan) << " has incompatible behavior";
      continue;
    }
#endif  // CUDNN_VERSION >= 8200
    plans.push_back(std::move(plan));
    if (max_workspace)
      *max_workspace = std::max(*max_workspace, static_cast<size_t>(workspace));
  }
  return plans;
}

#if !defined(__CUDACC__)  // Can be removed when CUDA 10 support is dropped.

Sampler MakeAvgSampler(size_t n, float max_cutoff_msec, size_t warmups) {
  size_t warmups_performed = 0;
  size_t k                 = 0;
  float s                  = 0.0f;
  if (n < 1)
    n = 1;

  return [n, max_cutoff_msec, warmups, warmups_performed, k, s](float x) mutable {
    if (warmups_performed < warmups && x < max_cutoff_msec) {
      warmups_performed++;
    } else {
      // Add this sample to the average calculation
      s += x;
      k++;
    }
    bool keep_going = k < n && x < max_cutoff_msec;
    return keep_going ? std::nullopt : std::optional(s / k);
  };
}

std::vector<FindResult> FindTopPlans(std::vector<Descriptor>&& plans,
                                     size_t max_results,
                                     cudnnHandle_t handle,
                                     const Descriptor& var_pack,
                                     Sampler sampler) {
  // We're about to perform kernel timings, so we need to quiet the system by grabbing
  // the Storage lock.  Concurrent cudaMalloc's can disrupt the accurate timing
  // measurements of the algos, and can prevent the cuda driver's proper freeing
  // of temporary workspace allocations.  Grabbing the lock might also
  // impede other threads from launching work on the GPU.
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  std::array<cudaEvent_t, 2> ev;
  for (auto& ee : ev)
    CUDA_CALL(cudaEventCreate(&ee));
  auto cmp = [](const FindResult& lhs, const FindResult& rhs) { return lhs.time < rhs.time; };
  cudaStream_t stream{};
  CUDNN_CALL(cudnnGetStream(handle, &stream));
  std::vector<FindResult> h;
  for (size_t i = 0; i < plans.size(); ++i) {
    auto&& plan = plans[i];
    // Make a copy of the unused sampler for each plan's timing.  Timed warm-up
    // runs are handled by the sampler to enable early loop exit for slow kernels.
    auto sampler_copy = sampler;
    for (;;) {
      CUDA_CALL(cudaEventRecord(ev[0], stream));
      CUDNN_CALL(cudnnBackendExecute(handle, plan.get(), var_pack.get()));
      CUDA_CALL(cudaEventRecord(ev[1], stream));
      CUDA_CALL(cudaEventSynchronize(ev[1]));
      float t = 0.0f;
      CUDA_CALL(cudaEventElapsedTime(&t, ev[0], ev[1]));
      if (auto r = sampler_copy(t); r) {
        auto time_to_record = r.value();
        if (h.size() == max_results) {
          if (time_to_record < h[0].time) {
            std::pop_heap(h.begin(), h.end(), cmp);
            h.back() = {std::move(plan), i, time_to_record};
            std::push_heap(h.begin(), h.end(), cmp);
          }
        } else {
          h.push_back({std::move(plan), i, time_to_record});
          std::push_heap(h.begin(), h.end(), cmp);
        }
        break;
      }
    }
  }
  for (auto& ee : ev)
    CUDA_CALL(cudaEventDestroy(ee));
  std::sort_heap(h.begin(), h.end(), cmp);
  return h;
}

#endif  // !defined(__CUDACC__)

std::string NoteStr(cudnnBackendNumericalNote_t note) {
  std::unordered_map<cudnnBackendNumericalNote_t, std::string> m{
      {CUDNN_NUMERICAL_NOTE_TENSOR_CORE, "tc"},
      {CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS, "dci"},
      {CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION, "rp"},
      {CUDNN_NUMERICAL_NOTE_FFT, "fft"},
      {CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC, "nd"},
      {CUDNN_NUMERICAL_NOTE_WINOGRAD, "w"},
  };
  auto it = m.find(note);
  return it != m.end() ? it->second : std::to_string(note);
}

std::string KnobStr(cudnnBackendKnobType_t knob) {
  std::unordered_map<cudnnBackendKnobType_t, std::string> m {
    {CUDNN_KNOB_TYPE_SPLIT_K, "split_k"}, {CUDNN_KNOB_TYPE_SWIZZLE, "swizzle"},
        {CUDNN_KNOB_TYPE_TILE_SIZE, "tile_size"}, {CUDNN_KNOB_TYPE_USE_TEX, "use_tex"},
        {CUDNN_KNOB_TYPE_EDGE, "edge"}, {CUDNN_KNOB_TYPE_KBLOCK, "kblock"},
        {CUDNN_KNOB_TYPE_LDGA, "ldga"}, {CUDNN_KNOB_TYPE_LDGB, "ldgb"},
        {CUDNN_KNOB_TYPE_CHUNK_K, "chunk_k"}, {CUDNN_KNOB_TYPE_SPLIT_H, "split_h"},
        {CUDNN_KNOB_TYPE_WINO_TILE, "wino_tile"}, {CUDNN_KNOB_TYPE_MULTIPLY, "multiply"},
        {CUDNN_KNOB_TYPE_SPLIT_K_BUF, "split_k_buf"}, {CUDNN_KNOB_TYPE_TILEK, "tilek"},
        {CUDNN_KNOB_TYPE_STAGES, "stages"}, {CUDNN_KNOB_TYPE_REDUCTION_MODE, "reduction_mode"},
        {CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE, "cta_split_k_mode"},
        {CUDNN_KNOB_TYPE_SPLIT_K_SLC, "split_k_slc"}, {CUDNN_KNOB_TYPE_IDX_MODE, "idx_mode"},
        {CUDNN_KNOB_TYPE_SLICED, "sliced"}, {CUDNN_KNOB_TYPE_SPLIT_RS, "split_rs"},
        {CUDNN_KNOB_TYPE_SINGLEBUFFER, "singlebuffer"}, {CUDNN_KNOB_TYPE_LDGC, "ldgc"},
        {CUDNN_KNOB_TYPE_SPECFILT, "specfilt"},
#if CUDNN_VERSION >= 8100
        {CUDNN_KNOB_TYPE_KERNEL_CFG, "kernel_cfg"},
#endif  // CUDNN_VERSION >= 8100
  };
  auto it = m.find(knob);
  return it != m.end() ? it->second : std::to_string(knob);
}

std::string PlanStr(const Descriptor& plan) {
  auto wks = GetAttr<int64_t>(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
  auto cfg =
      GetAttr(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
  auto engine     = GetAttr(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_BACKEND_ENGINE_DESCRIPTOR);
  auto engine_idx = GetAttr<int64_t>(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX);
  std::ostringstream ss;
  ss << "eng:" << engine_idx << " wksp:" << wks;
  auto notes = GetSomeAttrs<cudnnBackendNumericalNote_t>(
      CUDNN_NUMERICAL_NOTE_TYPE_COUNT, engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE);
  for (auto note : notes)
    ss << " " << NoteStr(note);
  auto choices = GetSomeAttrs(CUDNN_KNOB_TYPE_COUNTS,
                              cfg,
                              CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                              CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
  for (const auto& choice : choices) {
    auto type = GetAttr<cudnnBackendKnobType_t>(choice, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);
    auto val  = GetAttr<int64_t>(choice, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
    ss << " " << KnobStr(type) << ":" << val;
  }
  return ss.str();
}

}  // namespace cudnn_cxx
}  // namespace mxnet

#endif  // MXNET_USE_CUDNN == 1
