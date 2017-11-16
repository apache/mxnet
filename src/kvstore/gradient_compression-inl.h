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
 * \file gradient_compression-inl.h
 * \author Rahul Huilgol
 * \brief Declares and defines functions used to quantize and dequantize data
 */
#ifndef MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_
#define MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_

#include <vector>
#include "../operator/mxnet_op.h"

namespace mxnet {
namespace kvstore {

// these gpu functions are defined in gradient_compression.cu
void Quantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                      const float threshold);
void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu> *s, const std::vector<mxnet::TBlob> &inputs,
                        const float threshold);

struct quantize_2bit {
  MSHADOW_XINLINE static void Map(int out_block_id,
                                  int original_size,
                                  float *out,
                                  float *grad,
                                  float *residual,
                                  const float neg_threshold,
                                  const float pos_threshold) {
    // this block contains the compressed representation of
    // upto 16 values starting from out_block_id*16
    float *compr_block = out + out_block_id;
    // init to 0
    *compr_block = 0;
    // start and end are indices in original grad array
    const int start = out_block_id << 4;
    const int end = (start + 16 <= original_size) ? start + 16 : original_size;
    // cast as char* to manipulate bits of float addresses
    char *block_ptr = reinterpret_cast < char * > (compr_block);
    // masks to set bits when value meets pos_threshold
    // 0xc0 is mask when value is to be represented by the first two bits in a char*
    // 0xc0 means first two bits are set to 11
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    // masks to set bits when value meets neg_threshold
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    for (int i = start; i < end; i++) {
      // adds offset to reach appropriate byte
      char *curr_byte = block_ptr + ((i - start) >> 2);
      // adds gradient to existing residual to get updated grad
      residual[i] += grad[i];
      if (residual[i] >= pos_threshold) {
        // set data to 11
        *curr_byte |= posbits[(i & 3)];
        // reduce residual by pos_threshold
        residual[i] -= pos_threshold;
      } else if (residual[i] <= neg_threshold) {
        // set data to 10
        *curr_byte |= negbits[(i & 3)];
        residual[i] -= neg_threshold;
      }
    }
  }
};

template<typename xpu>
void Quantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                              const float threshold) {
  mxnet::op::mxnet_op::Kernel<quantize_2bit, xpu>
    ::Launch(s,
            inputs[2].Size(),         // compressed array size
            inputs[0].Size(),         // original size
            inputs[2].dptr<float>(),  // compressed array
            inputs[0].dptr<float>(),  // original array
            inputs[1].dptr<float>(),  // residual array
            -1 *threshold,            // negative threshold
            threshold);               // positive threshold
}

struct dequantize_2bit {
  MSHADOW_XINLINE static void Map(int i,
                                  float *out,
                                  float *in,
                                  const float neg_threshold,
                                  const float pos_threshold) {
    // get position of dequantized value to fill
    float *outval = out + i;
    // gets byte which holds quantized value for this position
    char *ch_ptr = reinterpret_cast<char *>(in + (i >> 4));
    ch_ptr += ((i & 15) >> 2);
    // masks used to quantize data
    const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
    // col denotes which two bits of a byte are set for this value
    // col=0 implies first two bits, col=3 implies last two bits,...
    const int col = i & 3;
    const uint8_t mask = posbits[col];
    const uint8_t negmask = negbits[col];
    const uint8_t masked = *ch_ptr & mask;
    if (masked == mask) {
      *outval = pos_threshold;
    } else if (masked == negmask) {
      // use posbits for mask as posbits are both 1s
      // then compare masked with negbits to see if only negbits were set
      *outval = neg_threshold;
    } else {
      *outval = 0;
    }
  }
};

template<typename xpu>
void Dequantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                                const float threshold) {
  mxnet::op::mxnet_op::Kernel<dequantize_2bit, xpu>
  ::Launch(s,
          inputs[1].Size(),         // original size
          inputs[1].dptr<float>(),  // out array
          inputs[0].dptr<float>(),  // compressed array
          -1 *threshold,            // negative threshold
          threshold);               // positive threshold
}

inline void Quantize2BitImpl(mshadow::Stream<mshadow::cpu> *s,
                             const std::vector<mxnet::TBlob> &inputs,
                             const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}

inline void Dequantize2BitImpl(mshadow::Stream<mshadow::cpu> *s,
                               const std::vector<mxnet::TBlob> &inputs,
                               const float threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}
}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_GRADIENT_COMPRESSION_INL_H_
