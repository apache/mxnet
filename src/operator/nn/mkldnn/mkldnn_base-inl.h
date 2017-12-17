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

/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_base-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*         zhengda1936@gmail.com
*
*******************************************************************************/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>
#include "mkldnn.hpp"
#include "mxnet/ndarray.h"
#include "mxnet/resource.h"
#include "mxnet/op_attr_types.h"
using namespace mkldnn;
namespace mxnet {
extern bool EnableMkldnnWarnGenerated();
// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine {
 public:
  static CpuEngine *Get() {
    // I's thread-safe in C++11.
    static thread_local CpuEngine myInstance;
    return &myInstance;
  }
  CpuEngine(CpuEngine const &) = delete;             // Copy construct
  CpuEngine(CpuEngine &&) = delete;                  // Move construct
  CpuEngine &operator=(CpuEngine const &) = delete;  // Copy assign
  CpuEngine &operator=(CpuEngine &&) = delete;       // Move assign

  mkldnn::engine &get_engine() { return _cpu_engine; }

 protected:
  CpuEngine() : _cpu_engine(mkldnn::engine::cpu, 0) {}
  ~CpuEngine() {}

 private:
  mkldnn::engine _cpu_engine;
};

// type enumerator
template <typename T>
struct data_type_enum {};

template <>
struct data_type_enum<float> {
  enum { type = mkldnn::memory::data_type::f32 };
};

template <>
struct data_type_enum<int32_t> {
  enum { type = mkldnn::memory::data_type::s32 };
};

template <>
struct data_type_enum<int16_t> {
  enum { type = mkldnn::memory::data_type::s16 };
};

template <>
struct data_type_enum<int8_t> {
  enum { type = mkldnn::memory::data_type::s8 };
};

template <>
struct data_type_enum<uint8_t> {
  enum { type = mkldnn::memory::data_type::u8 };
};

static inline bool SupportMKLDNNArray(int dtype, const TShape &shape) {
  int ndim = shape.ndim();
  bool support = ndim == 1 || ndim == 2 || ndim == 4;
  support = support && (dtype == mshadow::kFloat32 || dtype == mshadow::kInt32
                        || dtype == mshadow::kInt8 || dtype == mshadow::kUint8);
  return support;
}

static inline bool SupportStorageMKLDNN(int stype) {
  return stype == kMKLDNNStorage || stype == kDefaultStorage;
}

static inline bool SupportMKLDNN(int dtype, const TShape &shape) {
  int ndim = shape.ndim();
  return dtype == mshadow::kFloat32 && (ndim == 1 || ndim == 2 || ndim == 4);
}

static inline bool SupportMKLDNN(const NDArray &input) {
  return SupportMKLDNN(input.dtype(), input.shape())
      && SupportStorageMKLDNN(input.storage_type());
}

static inline bool SupportMKLDNNConv(const NDArray &input) {
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
}

static int GetTypeSize(int dtype) {
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    return sizeof(DType);
  });
  return -1;
}

static inline size_t GetArraySize(const NDArray &arr) {
  return arr.shape().Size() * GetTypeSize(arr.dtype());
}

static inline mkldnn::memory::data_type get_mkldnn_type(int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return mkldnn::memory::data_type::f32;
    case mshadow::kInt32:
      return mkldnn::memory::data_type::s32;
    case mshadow::kInt8:
      return mkldnn::memory::data_type::s8;
    case mshadow::kUint8:
      return mkldnn::memory::data_type::u8;
    default:
      LOG(FATAL) << "unknown type for MKLDNN";
      return mkldnn::memory::data_type::data_undef;
  }
}

inline static mkldnn::memory::desc GetMemDesc(const NDArray &arr, int ndim) {
  mkldnn::memory::dims dims(ndim);
  for (size_t i = 0; i < dims.size(); i++) dims[i] = arr.shape()[i];
  return mkldnn::memory::desc{dims, get_mkldnn_type(arr.dtype()),
                              mkldnn::memory::format::any};
}

inline static mkldnn::memory::desc GetMemDesc(const NDArray &arr) {
  return GetMemDesc(arr, arr.shape().ndim());
}

inline static mkldnn::memory::desc GetWeightDesc(const NDArray &arr,
                                                 int num_groups) {
  if (num_groups == 1) {
    return GetMemDesc(arr);
  } else {
    CHECK_EQ(arr.shape().ndim(), 4U);
    mkldnn::memory::dims tz = mkldnn::memory::dims{ num_groups,
      static_cast<int>(arr.shape()[0] / num_groups),
      static_cast<int>(arr.shape()[1]),
      static_cast<int>(arr.shape()[2]),
      static_cast<int>(arr.shape()[3])};
    return mkldnn::memory::desc{tz, get_mkldnn_type(arr.dtype()),
                                mkldnn::memory::format::any};
  }
}

typedef std::shared_ptr<mkldnn::memory> mkldnn_mem_ptr;
typedef std::shared_ptr<const mkldnn::memory> mkldnn_mem_const_ptr;

class TmpMemMgr {
  // This points to the memory buffer where we can allocate temp memory.
  char *curr_mem;
  // The total size of the temp memory.
  size_t mem_size;
  // This contains the current available memory size.
  size_t curr_size;
  // This estimate the required temp memory size in an operator.
  size_t est_size;
  const size_t alignment = 4096;

 public:
  static TmpMemMgr *Get() {
    static thread_local TmpMemMgr mgr;
    return &mgr;
  }

  TmpMemMgr() {
    Reset();
    est_size = 0;
    mem_size = 0;
  }

  void Reset() {
    curr_mem = nullptr;
    curr_size = 0;
    // We don't reset est_size and mem_size because est_size contains the
    // estimated temp memory size from the last run and mem_size contains the
    // memroy size allocated in the last run.
  }

  void Init(const Resource &r) {
    // If the last time, if we estimate that we need more memory, we should the
    // larger memory size.
    mem_size = std::max(mem_size, est_size);
    if (mem_size > 0) {
      // Let's allocate some extra memory. If we don't use some of them all the time,
      // the OS won't physically allocate pages for them any way.
      this->curr_size = mem_size * 2;
      this->curr_mem = static_cast<char *>(r.get_host_space_internal(this->curr_size));
    }
    // reset est_size, so we can start to estimate the temp memory size.
    this->est_size = 0;
  }

  mkldnn::memory *Alloc(const mkldnn::memory::primitive_desc &pd);
};

class MKLDNNStream {
  std::vector<mkldnn::primitive> net;
  // Here we hold all memory related to the operators in the stream.
  std::vector<std::shared_ptr<const mkldnn::memory> > mem_holder;

 public:
  static MKLDNNStream *Get() {
    static thread_local MKLDNNStream stream;
    return &stream;
  }

  void RegisterPrim(const mkldnn::primitive &prim) { net.push_back(prim); }

  void RegisterMem(std::shared_ptr<const mkldnn::memory> mem) {
    mem_holder.push_back(mem);
  }

  void Submit() {
    if (!net.empty())
      mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
    net.clear();
    mem_holder.clear();
    TmpMemMgr::Get()->Reset();
  }
};

enum OutDataOp {
  Noop,
  CopyBack,
  AddBack,
};

typedef std::pair<OutDataOp, mkldnn::memory *> mkldnn_output_t;

mkldnn_output_t CreateMKLDNNMem(const NDArray &arr,
                                const mkldnn::memory::primitive_desc &desc,
                                OpReqType req);
void CommitOutput(const NDArray &arr, const mkldnn_output_t &res);
const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::memory::primitive_desc &target_pd,
                                 int num_groups);
const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::engine &engine,
                                 int num_groups = 1);

mkldnn_memory_format_t GetDefaultFormat(mkldnn::memory::desc desc);
mkldnn::memory::primitive_desc GetPrimitiveDesc(mkldnn::memory::primitive_desc pd,
                                                mkldnn_memory_format_t format);


}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_
