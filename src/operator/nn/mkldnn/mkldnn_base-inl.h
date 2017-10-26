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

#ifndef MXNET_OPERATOR_MKL_MKLDNN_BASE_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_BASE_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <string>
#include <vector>
#include <iterator>
#include "mkldnn.hpp"

namespace mxnet {
extern bool EnableMkldnnWarnGenerated();
// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine {
 public:
    static CpuEngine & Instance() {
        // I's thread-safe in C++11.
        static thread_local CpuEngine myInstance;
        return myInstance;
    }
    CpuEngine(CpuEngine const&) = delete;             // Copy construct
    CpuEngine(CpuEngine&&) = delete;                  // Move construct
    CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
    CpuEngine& operator=(CpuEngine &&) = delete;      // Move assign

    mkldnn::engine & get_engine() { return _cpu_engine; }
 protected:
    CpuEngine() : _cpu_engine(mkldnn::engine::cpu, 0) {}
    ~CpuEngine() {}
 private:
    mkldnn::engine _cpu_engine;
};

// type enumerator
template<typename T>
struct data_type_enum {};

template<>
struct data_type_enum<float> {
    enum { type = mkldnn::memory::data_type::f32 };
};

template<>
struct data_type_enum<int32_t> {
    enum { type = mkldnn::memory::data_type::s32 };
};

template<>
struct data_type_enum<int16_t> {
    enum { type = mkldnn::memory::data_type::s16 };
};

template<>
struct data_type_enum<int8_t> {
    enum { type = mkldnn::memory::data_type::s8 };
};

template<>
struct data_type_enum<uint8_t> {
    enum { type = mkldnn::memory::data_type::u8 };
};

inline static std::shared_ptr<const mkldnn::memory> GetWeights(const NDArray &arr,
    const mkldnn::engine &engine, int num_groups = 1) {
  if (arr.shape().ndim() == 2) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{(int) arr.shape()[0],
      (int) arr.shape()[1]};
    mkldnn::memory::desc md = mkldnn::memory::desc{tz, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::oi};
    mkldnn::memory::primitive_desc pd = mkldnn::memory::primitive_desc{md, engine};
    std::vector<mkldnn::primitive> net;
    return arr.GetMKLDNNData(pd, net);
  }
  else if (arr.shape().ndim() == 4 && num_groups == 1) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{(int) arr.shape()[0],
      (int) arr.shape()[1], (int) arr.shape()[2], (int) arr.shape()[3]};
    mkldnn::memory::desc md = mkldnn::memory::desc{tz, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::oihw};
    mkldnn::memory::primitive_desc pd = mkldnn::memory::primitive_desc{md, engine};
    std::vector<mkldnn::primitive> net;
    return arr.GetMKLDNNData(pd, net);
  }
  else if (arr.shape().ndim() == 4) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{num_groups, (int) arr.shape()[0] / num_groups,
      (int) arr.shape()[1], (int) arr.shape()[2], (int) arr.shape()[3]};
    mkldnn::memory::desc md = mkldnn::memory::desc{tz, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::goihw};
    mkldnn::memory::primitive_desc pd = mkldnn::memory::primitive_desc{md, engine};
    std::vector<mkldnn::primitive> net;
    return arr.GetMKLDNNData(pd, net);
  }
  else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
    return nullptr;
  }
}

}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_MKL_MKLDNN_BASE_INL_H_
