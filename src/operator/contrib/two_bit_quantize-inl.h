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
 * \file two_bit_quantize-inl.h
 * \brief implementation of quantize_2bit operation
 * \author Chao Ma, Rahul Huilgol
 */

#ifndef MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_INL_H_
#define MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_INL_H_
#include <chrono>
#include <vector>
#include <limits>
#include <cmath>
#include <mxnet/operator_util.h>
#include <mxnet/c_api.h>
#include "../operator_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct init_mem_2bit {
  // Initialize output array
  MSHADOW_XINLINE static void Map(int i, float* out) {
    *(out+i) = 0;
  }
};

struct TwoBitParam : public dmlc::Parameter<TwoBitParam> {
  float threshold;
  DMLC_DECLARE_PARAMETER(TwoBitParam) {
    DMLC_DECLARE_FIELD(threshold)
      .set_default(0.5)
      .describe("Threshold to quantize values. "
                  "Must be greater than 0");
  }
};

template<typename xpu>
void Create2BitArrayCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  // For now, this method can only compress the float data
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // Init the memory of output to 0x00000000
  mxnet_op::Kernel<init_mem_2bit, xpu>::Launch(s, outputs[0].Size(),
                              outputs[0].dptr<float>());  // compressed array

}

inline bool Create2BitArrayShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  // 0. input array
  CHECK_EQ(in_attrs->size(), 1U);
  // 0. output array
  CHECK_EQ(out_attrs->size(), 1U);
  // check input
  CHECK(!shape_is_none(in_attrs->at(0)));
  // output
  int shape = in_attrs->at(0).Size() % 16 == 0 ?
                    in_attrs->at(0).Size() / 16 :
                    in_attrs->at(0).Size() / 16 + 1;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape{shape});
  return true;
}

inline bool Create2BitArrayType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  // 0. input array
  CHECK_EQ(in_attrs->size(), 1U);
  // 0. output array
  CHECK_EQ(out_attrs->size(), 1U);
  // check input
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`create_2bit_` only supports float32 input for now";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return true;
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
void Quantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<TBlob>& inputs, const float threshold) {
  mxnet_op::Kernel<quantize_2bit, xpu>::Launch(s, inputs[2].Size(), // compressed array size
                                               inputs[0].Size(),    // original size
                                               inputs[2].dptr<float>(),   // compressed array
                                               inputs[0].dptr<float>(),     // original array
                                               inputs[1].dptr<float>(),     // residual array
                                               -1 * threshold,            // negative threshold
                                               threshold);              // positive threshold
}

inline void Quantize2BitImpl(mshadow::Stream<cpu>* s, const std::vector<TBlob>& inputs, const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}

#ifndef __CUDACC__
void Quantize2BitImpl(mshadow::Stream<mshadow::gpu>* s, const std::vector<TBlob>& inputs,
                      const float threshold);
#else
template<typename xpu>
inline void Quantize2BitImpl(mshadow::Stream<gpu>* s, const std::vector<TBlob>& inputs,
                             const float threshold) {
  Quantize2BitKernelLaunch(s, inputs, threshold);
}
#endif


// this function has been defined as quantize_2bit operator
template<typename xpu>
void Quantize2BitCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TwoBitParam& param = nnvm::get<TwoBitParam>(attrs.parsed);
  Quantize2BitImpl(s, inputs, param.threshold);
}

inline bool Quantize2BitShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  // 0. input array
  // 1. residual array
  // 2. compressed array
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK(!shape_is_none(in_attrs->at(0)));
  CHECK(!shape_is_none(in_attrs->at(1)));
  CHECK_EQ(in_attrs->at(0).Size(),
           in_attrs->at(1).Size());
  int shape = in_attrs->at(0).Size() % 16 == 0 ?
                    in_attrs->at(0).Size() / 16 :
                    in_attrs->at(0).Size() / 16 + 1;
  CHECK_EQ(in_attrs->at(2).Size(), shape)
    << "The size of output array is not equal to "
    << "the size of compressed array";
  return true;
}

inline bool Quantize2BitType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  // 0. input array
  // 1. residual array
  // 2. compressed array
  CHECK_EQ(in_attrs->size(), 3U);
  // check input
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`quantize_2bit_` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "`quantize_2bit_` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32)
    << "`quantize_2bit_` only supports float32 input for now";
  return true;
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
void Dequantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<TBlob>& inputs, const float threshold) {
  mxnet::op::mxnet_op::Kernel<dequantize_2bit, xpu>::Launch(s, inputs[1].Size(),  // original size
                                                            inputs[1].dptr<float>(),        // out array
                                                            inputs[0].dptr<float>(),      // compressed array
                                                            -1*threshold,     // negative threshold
                                                            threshold);       // positive threshold
}

inline void Dequantize2BitImpl(mshadow::Stream<cpu>* s, const std::vector<TBlob>& inputs, const float threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}

#ifndef __CUDACC__
void Dequantize2BitImpl(mshadow::Stream<mshadow::gpu>* s, const std::vector<TBlob>& inputs,
                  const float threshold);
#else
template<typename xpu>
inline void Dequantize2BitImpl(mshadow::Stream<gpu>* s, const std::vector<TBlob>& inputs,
                             const float threshold) {
  Dequantize2BitKernelLaunch(s, inputs, threshold);
}
#endif

template<typename xpu>
void Dequantize2BitCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TwoBitParam& param = nnvm::get<TwoBitParam>(attrs.parsed);
  Dequantize2BitImpl(s, inputs, param.threshold);
}

inline bool Dequantize2BitShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape> *in_attrs,
                                std::vector<TShape> *out_attrs) {
  // 0. compressed array
  // 1. original array
  CHECK_EQ(in_attrs->size(), 2U);
  // No output
  CHECK_EQ(out_attrs->size(), 0U);
  // check input
  CHECK(!shape_is_none(in_attrs->at(0)));
  CHECK(!shape_is_none(in_attrs->at(1)));
  // TODO(huilgolr) check
  CHECK_LE(in_attrs->at(1).Size(),
           in_attrs->at(0).Size()*16)
    << "The shape of the second input array are "
    << "not equal to the original array.";
  return true;
}

inline bool Dequantize2BitType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  // 0. compressed array
  // 1. original array
  CHECK_EQ(in_attrs->size(), 2U);
  // check input
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`dequantize_2bit_` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "`dequantize_2bit_` only supports float32 input for now";
  return true;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_INL_H_
