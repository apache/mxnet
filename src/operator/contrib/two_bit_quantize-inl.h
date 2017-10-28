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
  */
#ifndef MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_INL_H_
#define MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_INL_H_
#include <chrono>
#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include <cmath>
#include <mxnet/c_api.h>
#include "ps/ps.h"

namespace mxnet {
namespace op {

struct init_mem_2bit {
  // Initialize output array
  MSHADOW_XINLINE static void Map(int i, float* out) {
    *(out+i) = 0;
  }
};

struct TwoBitParam : public dmlc::Parameter<TwoBitParam> {
  float pos_threshold, neg_threshold;
  DMLC_DECLARE_PARAMETER(TwoBitParam) {
    DMLC_DECLARE_FIELD(neg_threshold)
      .set_default(-0.1)
      .describe("Threshold to quantize negative values. "
                  "Has to be less than 0");
    DMLC_DECLARE_FIELD(pos_threshold)
      .set_default(0.1)
      .describe("Threshold to quantize positive values. "
                  "Has to be greater than 0");
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
    const int posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const int negbits[] = {0x80, 0x20, 0x08, 0x02};

    char* curr_byte = block_ptr;
    for (int i = start; i < end && i < original_size; i++) {
      // // adds 1 when i-start divisible by 4
      curr_byte += ((i - start) & 3);
      residual[i] += grad[i];
      if (residual[i] >= pos_threshold) {
        residual[i] -= pos_threshold;
        // set data to 11
        *curr_byte |= posbits[(i & 3)];
      } else if (residual[i] <= neg_threshold) {
        residual[i] -= neg_threshold;
        // set data to 10
        *curr_byte |= negbits[(i & 3)];
      }
    }
  }
};

template<typename xpu>
void Quantize2BitImplMShadow(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                      const float neg_threshold, const float pos_threshold) {

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
// compress the data and calculate new residual across all
  mxnet_op::Kernel<quantize_2bit, xpu>::Launch(s, inputs[2].Size(), // compressed array
                                               inputs[0].Size(),
                                               inputs[2].dptr<float>(),   // compressed array
                                               inputs[0].dptr<float>(),     // input array
                                               inputs[1].dptr<float>(),     // residual array
                                               neg_threshold,               // negative threshold
                                               pos_threshold);              // positive threshold
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "quantizing " << inputs[0].Size() << " took " << dur << " ms" << std::endl;

  if (dur > 1000) {
    NDArray *n = new NDArray(inputs[0], 0);
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create("quant_data", "w"));
    mxnet::NDArray::Save(fo.get(), {*n}, {});
  }
}

// this function has been defined as quantize_2bit operator
template<typename xpu>
void Quantize2BitComputeMShadow(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TwoBitParam& param = nnvm::get<TwoBitParam>(attrs.parsed);
  Quantize2BitImplMShadow<xpu>(s, inputs, param.neg_threshold, param.pos_threshold);
}

template<typename xpu>
void Quantize2BitCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TwoBitParam& param = nnvm::get<TwoBitParam>(attrs.parsed);
  float neg_threshold = param.neg_threshold;
  float pos_threshold = param.pos_threshold;
  int original_size = inputs[0].Size();
  float *out = inputs[2].dptr<float>();
  float *grad = inputs[0].dptr<float>();
  float *residual = inputs[1].dptr<float>();
  for (int out_block_id=0; out_block_id<inputs[2].Size(); out_block_id++) {
    float *compr_block = out + out_block_id;
    // init to 0
    *compr_block = 0;
    // start and end are indices in original grad array
    int start = out_block_id << 4;
    int end = start + 16; // <= original_size) ? start + 16 : original_size;
    char *block_ptr = reinterpret_cast < char * > (compr_block);
    const int posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const int negbits[] = {0x80, 0x20, 0x08, 0x02};

    char *curr_byte = block_ptr;
    for (int i = start; i < end && i < original_size; i++) {
      // // adds 1 when i-start divisible by 4
      curr_byte += ((i - start) & 3);
      residual[i] += grad[i];
      if (residual[i] >= pos_threshold) {
        residual[i] -= pos_threshold;
        // set data to 11
        *curr_byte |= posbits[(i & 3)];
      } else if (residual[i] <= neg_threshold) {
        residual[i] -= neg_threshold;
        // set data to 10
        *curr_byte |= negbits[(i & 3)];
      }
    }
  }
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
  MSHADOW_XINLINE static void Map(int compr_block_id,
                                  int original_size,
                                  float *out,
                                  float *in,
                                  const float neg_threshold,
                                  const float pos_threshold) {

    int out_start_id = compr_block_id<<4;
    float* outval = out + out_start_id;
    char* ch_ptr = reinterpret_cast<char*>(in + compr_block_id);
    const int posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const int negbits[] = {0x80, 0x20, 0x08, 0x02};
    for (int i = out_start_id; (i < out_start_id + 16) && (i < original_size); ++i, ++outval ) {
      ch_ptr += !(i & 3);
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
 }
};

template<typename xpu>
void Dequantize2BitImplMShadow(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                        const float neg_threshold, const float pos_threshold) {
  // Can only decompress the float32 data

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  int original_size = inputs[1].Size();
  mxnet_op::Kernel<dequantize_2bit, xpu>::Launch(s, original_size/16,  // original size
                                                 original_size,
                                                 inputs[1].dptr<float>(),        // out array
                                                 inputs[0].dptr<float>(),      // compressed array
                                                 neg_threshold,     // negative threshold
                                                 pos_threshold);  // positive threshold
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  std::cout<<"dequantizing "<<original_size<< " took "<<dur<<" ms"<<std::endl;
}

template<typename xpu>
void Dequantize2BitCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int original_size = inputs[1].Size();
  float* out = inputs[1].dptr<float>();
  float* in = inputs[0].dptr<float>();
  for (int compr_block_id=0; compr_block_id<original_size/16; compr_block_id++) {
    int out_start_id = compr_block_id<<4;
    float* outval = out + out_start_id;
    char* ch_ptr = reinterpret_cast<char*>(in + compr_block_id);
    const int posbits[] = {0xc0, 0x30, 0x0c, 0x03};
    const int negbits[] = {0x80, 0x20, 0x08, 0x02};
    for (int i = out_start_id; (i < out_start_id + 16) && (i < original_size); ++i, ++outval ) {
      ch_ptr += !(i & 3);
      int col = i & 3;
      uint8_t mask = posbits[col];
      uint8_t negmask = negbits[col];
      uint8_t masked = *ch_ptr & mask;
      if ( masked == mask ) {
        *outval = 0.5;
      } // use posbits for mask as posbits are 11
        // compare with negbits
      else if ( masked == negmask ) {
        *outval = -0.5;
      } else {
        *outval = 0;
      }
    }
  }
}

template<typename xpu>
void Dequantize2BitComputeMShadow(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  Dequantize2BitImplMShadow<xpu>(s, inputs, 0.5, 0.5);
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
