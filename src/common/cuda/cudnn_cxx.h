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
 * \file cudnn_cxx.h
 * \brief Convenience utilities to make coding against cuDNN v8 API less verbose
 */
#ifndef MXNET_COMMON_CUDA_CUDNN_CXX_H_
#define MXNET_COMMON_CUDA_CUDNN_CXX_H_

#include <mxnet/base.h>
#if MXNET_USE_CUDNN == 1

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#if !defined(__CUDACC__)  // Can be removed when CUDA 10 support is dropped.
#include <optional>       // NOLINT(build/include_order)
#endif                    // !defined(__CUDACC__)

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils.h"

STATIC_ASSERT_CUDNN_VERSION_GE(8002);

namespace mxnet {
namespace cudnn_cxx {

struct DescriptorDestroyer {
  using pointer = cudnnBackendDescriptor_t;

  void operator()(cudnnBackendDescriptor_t desc) {
    CUDNN_CALL_NONFATAL(cudnnBackendDestroyDescriptor(desc));
  }
};

using Descriptor = std::unique_ptr<cudnnBackendDescriptor_t, DescriptorDestroyer>;

struct WeakDescriptor {
  cudnnBackendDescriptor_t desc = nullptr;

  explicit WeakDescriptor(const Descriptor& other) : desc(other.get()) {}
  cudnnBackendDescriptor_t get() const {
    return desc;
  }
};

template <typename T>
struct AttrType;

template <>
struct AttrType<int64_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_INT64;
};

template <>
struct AttrType<void*> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_VOID_PTR;
};

template <>
struct AttrType<float> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_FLOAT;
};

template <>
struct AttrType<double> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_DOUBLE;
};

template <>
struct AttrType<cudnnHandle_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_HANDLE;
};

template <>
struct AttrType<bool> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_BOOLEAN;
};

template <>
struct AttrType<cudnnDataType_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_DATA_TYPE;
};

template <>
struct AttrType<cudnnConvolutionMode_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_CONVOLUTION_MODE;
};

template <>
struct AttrType<cudnnNanPropagation_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_NAN_PROPOGATION;
};

template <>
struct AttrType<cudnnPointwiseMode_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_POINTWISE_MODE;
};

template <>
struct AttrType<cudnnBackendHeurMode_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_HEUR_MODE;
};

template <>
struct AttrType<cudnnBackendNumericalNote_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_NUMERICAL_NOTE;
};

#if CUDNN_VERSION >= 8100
template <>
struct AttrType<cudnnReduceTensorOp_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_REDUCTION_OPERATOR_TYPE;
};
#if CUDNN_VERSION >= 8200
template <>
struct AttrType<cudnnBackendBehaviorNote_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_BEHAVIOR_NOTE;
};
#endif  // CUDNN_VERSION >= 8200
#endif  // CUDNN_VERSION >= 8100

template <>
struct AttrType<cudnnBackendKnobType_t> {
  static constexpr cudnnBackendAttributeType_t type = CUDNN_TYPE_KNOB_TYPE;
};

void SetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name, const Descriptor& val);
void SetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name, const WeakDescriptor& val);
void SetAttr(const Descriptor& desc,
             cudnnBackendAttributeName_t name,
             const std::vector<Descriptor>& val);

template <typename T>
void SetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name, T val) {
  CUDNN_CALL(cudnnBackendSetAttribute(desc.get(), name, AttrType<T>::type, 1, &val));
}

template <typename T>
void SetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name, const std::vector<T>& val) {
  CUDNN_CALL(cudnnBackendSetAttribute(desc.get(), name, AttrType<T>::type, val.size(), val.data()));
}

template <typename T, size_t N>
void SetAttr(const Descriptor& desc,
             cudnnBackendAttributeName_t name,
             const std::array<T, N>& val) {
  CUDNN_CALL(cudnnBackendSetAttribute(desc.get(), name, AttrType<T>::type, val.size(), val.data()));
}

inline void SetAttrs(const Descriptor& desc) {}

template <typename T, typename... Attrs>
void SetAttrs(const Descriptor& desc, cudnnBackendAttributeName_t name, T&& val, Attrs&&... rest) {
  SetAttr(desc, name, std::forward<T>(val));
  SetAttrs(desc, std::forward<Attrs>(rest)...);
}

std::vector<cudnnBackendDescriptor_t> MakeRawDescriptors(size_t n,
                                                         cudnnBackendDescriptorType_t type);

Descriptor Make(cudnnBackendDescriptorType_t type);

template <typename... Attrs>
Descriptor Make(cudnnBackendDescriptorType_t type, Attrs&&... attrs) {
  auto desc = Make(type);
  SetAttrs(desc, std::forward<Attrs>(attrs)...);
  return desc;
}

template <typename... Attrs>
Descriptor MakeFinalized(cudnnBackendDescriptorType_t type, Attrs&&... attrs) {
  auto desc = Make(type, std::forward<Attrs>(attrs)...);
  CUDNN_CALL(cudnnBackendFinalize(desc.get()));
  return desc;
}

template <typename T>
T GetAttr(const Descriptor& desc, cudnnBackendAttributeName_t name) {
  T ret{};
  int64_t ret_count = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(desc.get(), name, AttrType<T>::type, 1, &ret_count, &ret));
  CHECK_EQ(ret_count, 1);
  return ret;
}

template <typename T>
std::vector<T> GetAllAttrs(const Descriptor& desc, cudnnBackendAttributeName_t name) {
  int64_t count = 0;
  CUDNN_CALL(cudnnBackendGetAttribute(desc.get(), name, AttrType<T>::type, 0, &count, nullptr));
  std::vector<T> ret(count);
  CUDNN_CALL(cudnnBackendGetAttribute(
      desc.get(), name, AttrType<T>::type, ret.size(), &count, ret.data()));
  return ret;
}

template <typename T>
std::vector<T> GetSomeAttrs(size_t max_n,
                            const Descriptor& desc,
                            cudnnBackendAttributeName_t name) {
  int64_t count = 0;
  std::vector<T> ret(max_n);
  CUDNN_CALL(cudnnBackendGetAttribute(
      desc.get(), name, AttrType<T>::type, ret.size(), &count, ret.data()));
  ret.resize(count);
  return ret;
}

Descriptor GetAttr(const Descriptor& desc,
                   cudnnBackendAttributeName_t name,
                   cudnnBackendDescriptorType_t type);

std::vector<Descriptor> GetAllAttrs(const Descriptor& desc,
                                    cudnnBackendAttributeName_t name,
                                    cudnnBackendDescriptorType_t type);

std::vector<Descriptor> GetSomeAttrs(size_t max_n,
                                     const Descriptor& desc,
                                     cudnnBackendAttributeName_t name,
                                     cudnnBackendDescriptorType_t type);

// Order sets layout, as a permutation of dims, with N,C,<spacial dims> being identity.
template <typename T>
std::vector<T> PackedStrides(const std::vector<size_t>& order, const std::vector<T>& dims) {
  CHECK_EQ(order.size(), dims.size());
  std::vector<T> ret(dims.size(), 1);
  for (size_t i = dims.size() - 1; i--;)
    ret[order[i]] = dims[order[i + 1]] * ret[order[i + 1]];
  return ret;
}

// Given an engine config's `notes`, return whether that config is compatible, i.e. does
// the config have all of the required notes and none of the notes that are being excluded.
template <typename Note>
inline bool IsCompatible(const std::vector<Note>& notes,
                         const std::vector<Note>& require_notes,
                         const std::vector<Note>& exclude_notes) {
  for (auto rn : require_notes) {
    auto it = std::find(notes.begin(), notes.end(), rn);
    if (it == notes.end())
      return false;
  }
  for (auto en : exclude_notes) {
    auto it = std::find(notes.begin(), notes.end(), en);
    if (it != notes.end())
      return false;
  }
  return true;
}

// Execution plans are returned in the order of cuDNN heurstics, i.e. from best to worst.
// - max_workspace is an out parameter - the maximum workspace requirement among returned plans,
//   may be nullptr if not needed.
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
                                 bool verbose_filter);

#if !defined(__CUDACC__)  // Can be removed when CUDA 10 support is dropped.

// Defines a sampling algorithm.
// Returns an aggregate value, to be used as a metric for time comparison, or std::nullopt to
// perform another time measurement.
using Sampler = std::function<std::optional<float>(float)>;

// Return a sampler that after `n` trials returns the average.
// Before tallying trials, `warmups` trials are first ignored.
// If ever a trial that exceeds `max_cutoff_msec` is encountered (even during warmup),
// that trial is tallied and the sampling ends with the then-current trial average.
Sampler MakeAvgSampler(size_t n, float max_cutoff_msec = 1000.0, size_t warmups = 1);

struct FindResult {
  Descriptor plan;
  size_t heur_i;
  float time;
};

// Executes and times the plans. The results are returned in the order from best to worst.
std::vector<FindResult> FindTopPlans(std::vector<Descriptor>&& plans,
                                     size_t max_results,
                                     cudnnHandle_t handle,
                                     const Descriptor& var_pack,
                                     Sampler sampler);
#endif  // !defined(__CUDACC__)

std::string PlanStr(const Descriptor& plan);

}  // namespace cudnn_cxx
}  // namespace mxnet

#endif  // MXNET_USE_CUDNN == 1

#endif  //  MXNET_COMMON_CUDA_CUDNN_CXX_H_
