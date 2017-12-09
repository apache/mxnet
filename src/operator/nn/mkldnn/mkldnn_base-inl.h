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
#include "mkldnn.hpp"
using namespace mkldnn;
namespace mxnet {
extern bool EnableMkldnnWarnGenerated();
// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine {
 public:
  static CpuEngine &Instance() {
    // I's thread-safe in C++11.
    static thread_local CpuEngine myInstance;
    return myInstance;
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

static inline bool SupportMKLDNN(int dtype, const TShape &shape) {
  int ndim = shape.ndim();
  return dtype == mshadow::kFloat32 && (ndim == 1 || ndim == 2 || ndim == 4);
}

static inline bool SupportMKLDNN(const NDArray &input) {
  return SupportMKLDNN(input.dtype(), input.shape());
}

static inline bool SupportMKLDNNConv(const NDArray &input) {
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
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

class MKLDNNStream {
  std::vector<mkldnn::primitive> net;
  // Here we hold all memory related to the operators in the stream.
  std::vector<mkldnn_mem_const_ptr> mem_holder;

 public:
  static MKLDNNStream &Instance() {
    static thread_local MKLDNNStream stream;
    return stream;
  }

  void RegisterPrim(const mkldnn::primitive &prim) { net.push_back(prim); }

  void RegisterMem(mkldnn_mem_const_ptr mem) { mem_holder.push_back(mem); }

  void Submit() {
    if (!net.empty())
      mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
    net.clear();
    mem_holder.clear();
  }
};

inline static mkldnn_mem_ptr CreateMKLDNNMem(
    const mkldnn::memory::primitive_desc &desc) {
  // TODO(zhengda) allocate memory more efficiently.
  std::shared_ptr<mkldnn::memory> ret(new mkldnn::memory(desc));
  MKLDNNStream::Instance().RegisterMem(ret);
  return ret;
}

enum OutDataOp {
  Noop,
  CopyBack,
  AddBack,
};

typedef std::pair<OutDataOp, mkldnn_mem_ptr> mkldnn_output_t;

static inline mkldnn_output_t CreateMKLDNNMem(
    const NDArray &arr, const mkldnn::memory::primitive_desc &desc,
    OpReqType req) {
  if (kAddTo == req) {
    return mkldnn_output_t(OutDataOp::AddBack, CreateMKLDNNMem(desc));
  } else {
    mkldnn_mem_ptr mem = const_cast<NDArray &>(arr).CreateMKLDNNData(desc);
    if (mem == nullptr)
      return mkldnn_output_t(OutDataOp::CopyBack, CreateMKLDNNMem(desc));
    else
      return mkldnn_output_t(OutDataOp::Noop, mem);
  }
}

namespace op {
void Sum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
         const mkldnn::memory &out);
}

static inline void CommitOutput(const NDArray &arr,
                                const mkldnn_output_t &res) {
  if (res.first == CopyBack) {
    const_cast<NDArray &>(arr).CopyFrom(*res.second);
  } else if (res.first == AddBack) {
    mkldnn_mem_const_ptr mem =
        arr.GetMKLDNNData(res.second->get_primitive_desc());
    CHECK(mem != nullptr);
    // We have to allocate new memory for the sum result.
    mkldnn_mem_ptr sum_res(
        new mkldnn::memory(res.second->get_primitive_desc()));
    MKLDNNStream::Instance().RegisterMem(sum_res);
    op::Sum(*res.second, *mem, *sum_res);
    const_cast<NDArray &>(arr).CopyFrom(*sum_res);
  }
}

inline static mkldnn_mem_const_ptr GetWeights(
    const NDArray &arr, const mkldnn::memory::primitive_desc &target_pd,
    int num_groups) {
  mkldnn_mem_const_ptr mem;
  mkldnn::memory::data_type type = get_mkldnn_type(arr.dtype());
  auto engine = CpuEngine::Instance().get_engine();
  if (arr.shape().ndim() == 2) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oi};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    mem = arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4 && num_groups == 1) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1]),
          static_cast<int>(arr.shape()[2]), static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    mem = arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{ num_groups,
      static_cast<int>(arr.shape()[0] / num_groups),
      static_cast<int>(arr.shape()[1]),
      static_cast<int>(arr.shape()[2]),
      static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::goihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    mem = arr.GetMKLDNNData(pd);
  } else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
    return nullptr;
  }
  if (mem->get_primitive_desc() == target_pd) return mem;

  std::shared_ptr<mkldnn::memory> ret = CreateMKLDNNMem(target_pd);
  MKLDNNStream::Instance().RegisterPrim(mkldnn::reorder(*mem, *ret));
  return ret;
}

inline static mkldnn_mem_const_ptr GetWeights(const NDArray &arr,
                                              const mkldnn::engine &engine,
                                              int num_groups = 1) {
  mkldnn::memory::data_type type = get_mkldnn_type(arr.dtype());
  if (arr.shape().ndim() == 2) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oi};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    return arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4 && num_groups == 1) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1]),
          static_cast<int>(arr.shape()[2]), static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    return arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{ num_groups,
      static_cast<int>(arr.shape()[0] / num_groups),
      static_cast<int>(arr.shape()[1]),
      static_cast<int>(arr.shape()[2]),
      static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::goihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    return arr.GetMKLDNNData(pd);
  } else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
    return nullptr;
  }
}

}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_
