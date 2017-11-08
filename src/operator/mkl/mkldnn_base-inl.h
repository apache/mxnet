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
        static CpuEngine myInstance;
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

// =====  MKLDNNStream =======================================
class MKLDNNStream {
 public:
    MKLDNNStream():_ready(false) { prepare(); }
    virtual ~MKLDNNStream() {}
    MKLDNNStream  &submit(std::vector<mkldnn::primitive> primitives) {
        _stream->submit(primitives); return *this;
    }
    bool wait(bool block = true) {
        _ready = false;
        bool res = _stream->wait(block);
        return res;
    }
    bool ready() { return _ready; }
    void prepare() {
        if (_ready == false) {
            _stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
        }
        _ready = true;
    }

 private:
    bool _ready;
    std::shared_ptr<mkldnn::stream> _stream;
};

// =====  StreamHolder =======================================
// singleton
class StreamHolder {
 public:
    static StreamHolder & Instance() {
        // I's thread-safe in C++11.
        static StreamHolder myInstance;
        return myInstance;
    }
    StreamHolder(StreamHolder const&) = delete;             // Copy construct
    StreamHolder(StreamHolder&&) = delete;                  // Move construct
    StreamHolder& operator=(StreamHolder const&) = delete;  // Copy assign
    StreamHolder& operator=(StreamHolder &&) = delete;      // Move assign

    std::shared_ptr<MKLDNNStream> get_stream();
    std::shared_ptr<MKLDNNStream> current_stream() { return _current_stream; }
    void prepare_mkldnn_stream(std::shared_ptr<MKLDNNStream> mkldnn_stream) {
        _current_stream = mkldnn_stream;
        _current_stream->prepare();
    }
 protected:
    StreamHolder() : _current_stream(NULL) {}
    ~StreamHolder() {}
 private:
    std::shared_ptr<MKLDNNStream> _current_stream;
};

// =====  MKLDNNLayer =======================================
template <typename Dtype>
class MKLDNNLayer {
 public:
    MKLDNNLayer() {}
    virtual ~MKLDNNLayer() {}
};

// =====  MKLDNNPrimitive =======================================
template <typename Dtype>
class MKLDNNPrimitive {
 public:
    MKLDNNPrimitive():aprimitive(NULL), mkldnn_stream(NULL) {}
    virtual ~MKLDNNPrimitive() {}
    void reset(mkldnn::primitive* pprimitive) { this->aprimitive.reset(pprimitive);}
    std::shared_ptr<mkldnn::primitive> aprimitive;
    std::shared_ptr<MKLDNNStream> mkldnn_stream;
    std::shared_ptr<MKLDNNStream> get_mkldnn_stream();
    std::shared_ptr<MKLDNNStream> submit();
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

}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_MKL_MKLDNN_BASE_INL_H_
