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
 * \file gc.cc
 * \brief Gradient compression for kvstore
 * \author Rahul Huilgol
 */
#include <mxnet/gc.h>
#include <sstream>
#include <vector>
// for get_rank
#include <ps/ps.h>
#include "./gc-inl.h"

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

void Gc::set_active() {
  active_ = true;
}

void Gc::set_inactive() {
  active_ = false;
}

bool Gc::get_active() {
  return active_;
}

CompressionType Gc::get_type() {
  return type_;
}

bool Gc::get_active_type() {
  if (active_) return type_;
  else return GC_NONE;
}

void Gc::increment_push(int key) {
//  if (!get_active()) {
    std::unordered_map<int, int>::const_iterator got = num_pushes_.find(key);
    if (got == num_pushes_.end()) {
      // first push is init, so not counting that
      num_pushes_[key] = 0;
    } else {
      num_pushes_[key] += 1;
    }
//    if(ps::MyRank()==0) std::cout<<"numpush for key "<<key<<" is "<<num_pushes_[key]<<std::endl;
    // if we see n+1 th push for any key, then we have waited for n pushes for all keys TODO(true?)
    // if delay is 0, this means we set active after second push (which is first non-init push)
    if (num_pushes_[key] > 0) {
//      if(ps::MyRank()==0) std::cout<<"would set active "<<std::endl;
//      set_active();
    }
//  } // else only needs to be added if we want to set GC inactive at some point after it turns on
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
    LOG(FATAL) << "Unsupported compression type";
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
  if(ps::MyRank()==0) std::cout<<from.shape().Size()<< " to "<<to->shape().Size()<<std::endl;
  CHECK(from.shape().ndim() != 0) << "source operands have zero dimension shape";
  int a = from.ctx().dev_mask();
  int b = to->ctx().dev_mask();
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
  CHECK(from.shape().ndim() != 0) << "source operands have zero dimension shape";
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

} // namespace kvstore
} // namespace mxnet

