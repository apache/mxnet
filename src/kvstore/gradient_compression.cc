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
 * \file gradient_compression.cc
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */

#include <sstream>
#include <vector>
#include "gradient_compression.h"
#include "gradient_compression-inl.h"

namespace mxnet {
namespace kvstore {

/*!
 * \brief Splits a string into smaller strings using char as delimiter
 * Example: "a,b,c,,d" is split into ["a","b","c","","d"]
 * \param s string to split
 * \param delim char to split string around
 * \param result container for tokens extracted after splitting
 */
template<typename Out>
void split(const std::string &s, const char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

Gc::Gc() {
  type_ = GC_NONE;
  active_ = false;
}

void Gc::SetParams(const std::string &compression_type, const float threshold) {
  if (compression_type == "2bit") {
    SetTwoBitCompression(threshold);
  }
}

void Gc::set_active(bool active) {
  active_ = active;
}

// note that this can be active when type is none, it denotes init is done for now
bool Gc::is_active() {
  return active_;
}

CompressionType Gc::get_type() {
  return type_;
}

void Gc::SetTwoBitCompression(const float threshold) {
  type_ = GC_TWO_BIT;
  threshold_ = threshold;
}

std::string Gc::EncodeParams() {
  std::string rval = std::to_string(type_);
  if (type_ == GC_TWO_BIT) {
    rval += "," + std::to_string(threshold_);
  }
  return rval;
}

void Gc::DecodeParams(const std::string &s) {
  std::vector<std::string> elems;
  split(s, ',', std::back_inserter(elems));
  type_ = static_cast<CompressionType>(stoi(elems[0]));
  if (elems.size() > 1) {
    if (!elems[1].empty()) {
      threshold_ = stof(elems[1]);
    }
  }
}

int Gc::GetCompressionFactor() {
  if (type_ == GC_TWO_BIT) {
    return 16;
  } else {
    LOG(FATAL) << "Unsupported compression type: " << type_;
    return 0;
  }
}

int64_t Gc::GetCompressedSize(const int64_t original_size) {
  const int bits = GetCompressionFactor();
  return ((original_size % bits == 0) ?
          original_size / bits :
          original_size / bits + 1);
}

void Gc::Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                  mxnet::NDArray *residual, const int priority) {
  CHECK(from.shape().ndim() != 0) << "source operand has zero dimension shape";
  CHECK(to->shape().ndim() != 0) << "destination operand has zero dimension shape";
  CHECK(residual->shape().ndim() != 0) << "residual operand has zero dimension shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  const float threshold = threshold_;
  if (type_ == GC_TWO_BIT) {
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, residual, threshold](mxnet::RunContext ctx) {
        std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
        Quantize2BitImpl(ctx.get_stream<mshadow::cpu>(), inputs, threshold);
      }, from.ctx(), {from.var()}, {to->var(), residual->var()},
      mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("QuantizeCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, residual, threshold](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), residual->data(), to->data()};
          Quantize2BitImpl(ctx.get_stream<mshadow::gpu>(), inputs, threshold);
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

void Gc::Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to, const int priority) {
  CHECK(from.shape().ndim() != 0) << "source operands has zero dimension shape";
  CHECK(to->shape().ndim() != 0) << "destination operand has zero dimension shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  const float threshold = threshold_;
  if (type_ == GC_TWO_BIT) {
    if (a == mshadow::cpu::kDevMask && b == mshadow::cpu::kDevMask) {
      mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
        std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
        Dequantize2BitImpl(ctx.get_stream<mshadow::cpu>(), inputs, threshold);
      }, from.ctx(), {from.var()}, {to->var()},
      mxnet::FnProperty::kNormal, priority, PROFILER_MESSAGE("DequantizeCPU"));
    } else {
#if MXNET_USE_CUDA
      if (a == mshadow::gpu::kDevMask && b == mshadow::gpu::kDevMask) {
        mxnet::Engine::Get()->PushSync([from, to, threshold](mxnet::RunContext ctx) {
          std::vector<mxnet::TBlob> inputs = {from.data(), to->data()};
          Dequantize2BitImpl(ctx.get_stream<mshadow::gpu>(), inputs, threshold);
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

}  // namespace kvstore
}  // namespace mxnet

