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
#include <operator/mxnet_op.h>

namespace mxnet {
  namespace kvstore {

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

Gc::Gc() {
  type_ = GC_NONE;
  active_ = false;
}

void Gc::SetParams(const std::string& compression_type, const float threshold) {
  if (compression_type == "2bit") {
    SetTwoBitCompression(threshold);
  }
}

void Gc::set_active() {
  active_ = true;
}

bool Gc::get_active() {
  return active_;
}

bool Gc::get_active_type() {
  if (active_) return type_;
  else return GC_NONE;
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

void Gc::DecodeParams(const std::string& s) {
  std::vector<std::string> elems;
  split(s, ',', std::back_inserter(elems));
  type_ = static_cast<CompressionType>(stoi(elems[0]));
  if (elems.size()>1) {
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

int64_t Gc::GetCompressedSize(const int64_t original_size){
  const int bits = GetCompressionFactor();
  return ((original_size % bits == 0) ?
           original_size  / bits :
           original_size  / bits + 1);
}

struct quantize_2bit {
  MSHADOW_XINLINE static void Map(int out_block_id,
                                  int original_size,
                                  float *out,
                                  float *grad,
                                  float *residual,
                                  const float neg_threshold,
                                  const float pos_threshold) {
    float* compr_block = out + out_block_id;
    // init to 0
    *compr_block = 0;
    // start and end are indices in original grad array
    int start = out_block_id << 4;
    int end = start + 16; // <= original_size) ? start + 16 : original_size;
    char* block_ptr = reinterpret_cast < char* > (compr_block);
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};

    for (int i = start; i < end && i < original_size; i++) {
      // // adds 1 when i-start divisible by 4
      char* curr_byte = block_ptr + ((i-start)>>2);
      residual[i] += grad[i];
      if (residual[i] >= pos_threshold) {
        residual[i] -= pos_threshold;
        // set data to 11
        *curr_byte |= posbits[(i & 3)];
//        std::cout<<"pos "<< std::to_string(i&3) << " " << std::bitset<8>(*curr_byte)<<std::endl;
      } else if (residual[i] <= neg_threshold) {
        residual[i] -= neg_threshold;
        // set data to 10
        *curr_byte |= negbits[(i & 3)];
//        std::cout<<"neg "<< std::to_string(i&3) << " " << std::bitset<8>(*curr_byte)<<std::endl;
      } else {
//        std::cout<<"0 "<< std::to_string(i&3) << " " << std::bitset<8>(*curr_byte)<<std::endl;
      }
    }
  }
};

template<typename xpu>
void Quantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob>& inputs, const float threshold) {
  mxnet::op::mxnet_op::Kernel<mxnet::kvstore::quantize_2bit, xpu>::Launch(s, inputs[2].Size(), // compressed array size
                                               inputs[0].Size(),    // original size
                                               inputs[2].dptr<float>(),   // compressed array
                                               inputs[0].dptr<float>(),     // original array
                                               inputs[1].dptr<float>(),     // residual array
                                               -1 * threshold,            // negative threshold
                                               threshold);              // positive threshold
}

void Quantize2BitImpl(mshadow::Stream<mshadow::cpu>* s, const std::vector<mxnet::TBlob>& inputs, const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}

struct dequantize_2bit {
  // Decompress
  MSHADOW_XINLINE static void Map(int i,
                                  float *out,
                                  float *in,
                                  const float neg_threshold,
                                  const float pos_threshold) {

    float* outval = out + i;
    char* ch_ptr = reinterpret_cast<char*>(in + (i>>4));

    ch_ptr += ((i & 15) >> 2 );
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    int col = i & 3;
    uint8_t mask = posbits[col];
    uint8_t negmask = negbits[col];
    uint8_t masked = *ch_ptr & mask;
    if ( masked == mask ) {
      *outval = pos_threshold;
    } // use posbits for mask as posbits are 11
      // compare with negbits
    else if ( masked == negmask ) {
      *outval = neg_threshold;
    } else {
      *outval = 0;
    }
  }
};

template<typename xpu>
void Dequantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob>& inputs, const float threshold) {

  mxnet::op::mxnet_op::Kernel<mxnet::kvstore::dequantize_2bit, xpu>::Launch(s, inputs[1].Size(),  // original size
                                                            inputs[1].dptr<float>(),        // out array
                                                            inputs[0].dptr<float>(),      // compressed array
                                                            -1*threshold,     // negative threshold
                                                            threshold);       // positive threshold
}

void Dequantize2BitImpl(mshadow::Stream<mshadow::cpu>* s, const std::vector<mxnet::TBlob>& inputs, const float threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}

void Quantize2BitImpl(mshadow::Stream<mshadow::gpu>* s, const std::vector<mxnet::TBlob>& inputs, const float threshold);
void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu>* s, const std::vector<mxnet::TBlob>& inputs, const float threshold);


    void Gc::Quantize(const mxnet::NDArray &from, mxnet::NDArray *to,
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

  }
}

