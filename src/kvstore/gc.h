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
 * \file gc.h
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#ifndef MXNET_KVSTORE_GC_H
#define MXNET_KVSTORE_GC_H
#include <string>
#include <sstream>
#include <vector>

#include"../../include/mxnet/ndarray.h"
#include "../../include/mxnet/base.h"
#include <mshadow/tensor.h>
#include "../operator/contrib/two_bit_quantize-inl.h"

// TODO check if it returns empty between two delims
template<typename Out>
void split(const std::string &s, const char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

enum CompressionType {
  GC_NONE, GC_TWO_BIT
};

class Gc {
  public:
    Gc() {
      type_ = GC_NONE;
      active_ = false;
    }

    virtual ~Gc() { }

    void SetParams(const std::string& compression_type, const float threshold) {
      if (compression_type == "2bit") {
        SetTwoBitCompression(threshold);
      }
    }

    void set_active() {
      active_ = true;
    }

    bool get_active_type() {
      if (active_) return type_;
      else return GC_NONE;
    }

    void SetTwoBitCompression(const float threshold) {
      type_ = GC_TWO_BIT;
      threshold_ = threshold;
    }

    std::string EncodeParams() {
      std::string rval = std::to_string(type_);
      if (type_ == GC_TWO_BIT) {
        rval += "," + std::to_string(threshold_);
      }
      return rval;
    }

    void DecodeParams(const std::string& s) {
      std::vector<std::string> elems;
      split(s, ',', std::back_inserter(elems));
      type_ = static_cast<CompressionType>(stoi(elems[0]));
      if (elems.size()>1) {
        if (!elems[1].empty()) {
          threshold_ = stof(elems[1]);
        }
      }
    }

    int GetCompressionFactor() {
      if (type_ == GC_TWO_BIT) {
        return 16;
      } else {
        LOG(FATAL) << "Unsupported compression type";
        return 0;
      }
    }

    int64_t GetCompressedSize(const int64_t original_size){
      const int bits = GetCompressionFactor();
      return ((original_size % bits == 0) ?
               original_size  / bits :
               original_size  / bits + 1);
    }

    void Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                  mxnet::NDArray *residual, const int priority) {
      CHECK(from.shape().ndim() != 0)
        << "source operands have zero dimension shape";
      // important: callback must always capture by value
      int a = from.ctx().dev_mask();
      int b = to->ctx().dev_mask();
      const float threshold = threshold_;
      if (type_ == GC_TWO_BIT) {
        if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
          mxnet::Engine::Get()->PushSync([from, to, residual, threshold](mxnet::RunContext ctx) {
            std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
            mxnet::op::Quantize2BitImpl(ctx.get_stream<mshadow::cpu>(), inputs, threshold);
            }, from.ctx(), {from.var()}, {to->var(), residual->var()},
            mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeCPU"));
        } else {
          #if MXNET_USE_CUDA
          if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
            mxnet::Engine::Get()->PushSync([from, to, residual, threshold](mxnet::RunContext ctx) {
              std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
              mxnet::op::Quantize2BitImpl(ctx.get_stream<mshadow::gpu>(), inputs, threshold);
              // Wait GPU kernel to complete
              ctx.get_stream<mshadow::gpu>()->Wait();
            }, from.ctx(), {from.var()}, {to->var(), residual->var()},
            mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeGPU"));
          } else {
            LOG(FATAL) << "unknown device mask";
          }
          #else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
          #endif
        }
      } else {
        LOG(FATAL) << "Unsupported quantization of type " << type_;
      }
    }

  void Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority) {
    CHECK(from.shape().ndim() != 0)
      << "source operands have zero dimension shape";
    // important: callback must always capture by value
    const int a = from.ctx().dev_mask();
    const int b = to->ctx().dev_mask();
    const float threshold = threshold_;
    if (type_ == GC_TWO_BIT) {
      if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
          mxnet::op::Dequantize2BitImpl(ctx.get_stream<mshadow::cpu>(), inputs, threshold);
          }, from.ctx(), {from.var()}, {to->var()},
          mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeCPU"));
      } else {
        #if MXNET_USE_CUDA
        if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
            mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
              std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
              mxnet::op::Dequantize2BitImpl(ctx.get_stream<mshadow::gpu>(), inputs, threshold);
              // Wait GPU kernel to complete
              ctx.get_stream<mshadow::gpu>()->Wait();
            }, from.ctx(), {from.var()}, {to->var()},
            mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeGPU"));
          } else {
            LOG(FATAL) << "unknown device mask";
          }
        #else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        #endif
      }
    } else {
      LOG(FATAL) << "Unsupported dequantization of type " << type_;
    }
  }

  CompressionType type_;

  bool active_;

  float threshold_ = 0;

private:

};


#endif //MXNET_KVSTORE_GC_H
