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

DMLC_REGISTER_PARAMETER(GradientCompressionParam);

GradientCompression::GradientCompression() {
  type_ = CompressionType::kNone;
}

void GradientCompression::SetParams(const std::vector<std::pair<std::string, std::string> >
                                    & kwargs) {
  GradientCompressionParam params;
  params.InitAllowUnknown(kwargs);
  CHECK_GT(params.threshold, 0) << "threshold must be greater than 0";
  if (params.type == "2bit") {
    SetTwoBitCompression(params.threshold);
  } else {
    LOG(FATAL) << "Unknown type for gradient compression " << params.type;
  }
}

CompressionType GradientCompression::get_type() {
  return type_;
}

std::string GradientCompression::get_type_str() {
  return std::to_string(static_cast<int>(type_));
}

void GradientCompression::SetTwoBitCompression(const float threshold) {
  type_ = CompressionType::kTwoBit;
  threshold_ = threshold;
}

std::string GradientCompression::EncodeParams() {
  using namespace std;  // to reduce length of next line
  string rval = get_type_str();
  if (type_ == CompressionType::kTwoBit) {
    rval += "," + to_string(threshold_);
  }
  return rval;
}

void GradientCompression::DecodeParams(const std::string &s) {
  std::vector<std::string> elems;
  split(s, ',', std::back_inserter(elems));
  type_ = static_cast<CompressionType>(stoi(elems[0]));
  if (elems.size() > 1) {
    if (!elems[1].empty()) {
      threshold_ = stof(elems[1]);
    }
  }
}

int GradientCompression::GetCompressionFactor() {
  if (type_ == CompressionType::kTwoBit) {
    return 16;
  } else {
    LOG(FATAL) << "Unsupported compression type: " << get_type_str();
    return 0;
  }
}

int64_t GradientCompression::GetCompressedSize(const int64_t original_size) {
  const int bits = GetCompressionFactor();
  return ((original_size % bits == 0) ?
          original_size / bits :
          original_size / bits + 1);
}

void GradientCompression::Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                  mxnet::NDArray *residual, const int priority) {
  CHECK(from.shape().ndim() != 0) << "source operand has zero dimension shape";
  CHECK(to->shape().ndim() != 0) << "destination operand has zero dimension shape";
  CHECK(residual->shape().ndim() != 0) << "residual operand has zero dimension shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  const float threshold = threshold_;
  if (type_ == CompressionType::kTwoBit) {
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
    LOG(FATAL) << "Unsupported quantization of type " << get_type_str();
  }
}

void GradientCompression::Dequantize(const mxnet::NDArray &from, mxnet::NDArray *to,
                                     const int priority) {
  CHECK(from.shape().ndim() != 0) << "source operands has zero dimension shape";
  CHECK(to->shape().ndim() != 0) << "destination operand has zero dimension shape";
  const int a = from.ctx().dev_mask();
  const int b = to->ctx().dev_mask();
  const float threshold = threshold_;
  if (type_ == CompressionType::kTwoBit) {
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
    LOG(FATAL) << "Unsupported dequantization of type " << get_type_str();
  }
}

}  // namespace kvstore
}  // namespace mxnet

