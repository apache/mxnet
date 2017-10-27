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

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include <cmath>
#include "ps/ps.h"

namespace mxnet {
namespace op {

// branchless
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}


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

//struct init_threshold_2bit {
//  MSHADOW_XINLINE static void Map(int server_id,
//                                  float *out,
//                                  const float neg_threshold,
//                                  const float pos_threshold,
//                                  ps::SArray<int> compr_sizes,
//                                  ps::SArray<int> orig_sizes) {
//    // i for each server
//    size_t curr_pos = 0;
//    for (int i=0; i<server_id; i++) {
//      curr_pos += compr_sizes[i];
//    }
//
//    // The first two elements in output are thresholds
//    // The third element is the original size of the array
//    out[curr_pos] = neg_threshold;
//    out[curr_pos + 1] = pos_threshold;
//    // TODO(huilgolr) check potential problem here?
//    out[curr_pos+2] = static_cast<float>(orig_sizes[server_id]);
//  }
//};

struct quantize_2bit {
  MSHADOW_XINLINE static void Map(int out_block_id,
//                                  std::unordered_set<int> meta_pos,
//                                  std::vector<int> cumulative_part_indices,
//                                  ps::SArray<int> compr_sizes,
//                                  ps::SArray<int> orig_sizes,
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
void Quantize2BitImpl(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                      ps::SArray<int> push_pskv_lens, ps::SArray<int> pull_pskv_lens,
                      const float neg_threshold, const float pos_threshold) {
  // Init threshold and original size
//  mxnet_op::Kernel<init_threshold_2bit, xpu>::Launch(s, push_pskv_lens.size(),
//                              inputs[2].dptr<float>(),   // compressed array (concat for all servers)
//                              neg_threshold, pos_threshold,
//                              push_pskv_lens, pull_pskv_lens);

//  std::unordered_set<int> meta_pos;
//  std::vector<int> cumulative_part_indices;
//  int cur_pos = 0;
//  int cum_index = 0;
//  for(int i=0; i<push_pskv_lens.size(); i++) {
//    meta_pos.insert(cur_pos);
//    meta_pos.insert(cur_pos+1);
//    meta_pos.insert(cur_pos+2);
//    cur_pos += push_pskv_lens[i];
//    cumulative_part_indices.push_back(cur_pos);
//  }

// Finally, compress the data and calculate new residual across all
  mxnet_op::Kernel<quantize_2bit, xpu>::Launch(s, inputs[2].Size(), // compressed array
                          inputs[0].Size(),
//                          meta_pos, cumulative_part_indices,
//                          push_pskv_lens,            // compressed sizes
//                          pull_pskv_lens,            // original sizes
                          inputs[2].dptr<float>(),   // compressed array
                          inputs[0].dptr<float>(),     // input array
                          inputs[1].dptr<float>(),     // residual array
                          neg_threshold,               // negative threshold
                          pos_threshold);              // positive threshold
}

template<typename xpu>
void Quantize2BitImplMShadow(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                      const float neg_threshold, const float pos_threshold) {
  // Init threshold and original size
//  mxnet_op::Kernel<init_threshold_2bit, xpu>::Launch(s, push_pskv_lens.size(),
//                              inputs[2].dptr<float>(),   // compressed array (concat for all servers)
//                              neg_threshold, pos_threshold,
//                              push_pskv_lens, pull_pskv_lens);

//  std::unordered_set<int> meta_pos;
//  std::vector<int> cumulative_part_indices;
//  int cur_pos = 0;
//  int cum_index = 0;
//  for(int i=0; i<push_pskv_lens.size(); i++) {
//    meta_pos.insert(cur_pos);
//    meta_pos.insert(cur_pos+1);
//    meta_pos.insert(cur_pos+2);
//    cur_pos += push_pskv_lens[i];
//    cumulative_part_indices.push_back(cur_pos);
//  }

// Finally, compress the data and calculate new residual across all
  mxnet_op::Kernel<quantize_2bit, xpu>::Launch(s, inputs[2].Size(), // compressed array
                                               inputs[0].Size(),
//                          meta_pos, cumulative_part_indices,
//                          push_pskv_lens,            // compressed sizes
//                          pull_pskv_lens,            // original sizes
                                               inputs[2].dptr<float>(),   // compressed array
                                               inputs[0].dptr<float>(),     // input array
                                               inputs[1].dptr<float>(),     // residual array
                                               neg_threshold,               // negative threshold
                                               pos_threshold);              // positive threshold
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

struct dequantize_2bit_all {
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


    // get row ptr
//    char* ch_ptr = (reinterpret_cast<char*>(in + (i >> 4)));
//    for (int i=0 )
//    + ((i & 15) >> 2);
//    const int posbits[] = {0xc0, 0x30, 0x0c, 0x03};
//    const int negbits[] = {0x80, 0x20, 0x08, 0x02};
//
//    int col = (i & 15) & 3;
//    if ( ((*ch_ptr) & posbits[col]) == posbits[col] ) {
//      out[i] = pos_threshold;
//    } // use posbits for mask as posbits are 11
//      // compare with negbits
//    else if ( ((*ch_ptr) & posbits[col]) == negbits[col] ) {
//      out[i] = neg_threshold;
//    } else {
//      out[i] = 0;
//    }
 }
};

template<typename xpu>
void Dequantize2BitImplMShadow(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                        const float neg_threshold, const float pos_threshold) {
  // Can only decompress the float32 data
  int original_size = inputs[1].Size();
  mxnet_op::Kernel<dequantize_2bit, xpu>::Launch(s, original_size/16,  // original size
                                                 original_size,
                                                 inputs[1].dptr<float>(),        // out array
                                                 inputs[0].dptr<float>(),      // compressed array
                                                 neg_threshold,     // negative threshold
                                                 pos_threshold);  // positive threshold

}
  template<typename xpu>
  void Dequantize2BitImpl(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                                 const float neg_threshold, const float pos_threshold) {
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
}

template<typename xpu>
void Dequantize2BitCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
//  Dequantize2BitImpl<xpu>(s, inputs, 0.5, 0.5);
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
