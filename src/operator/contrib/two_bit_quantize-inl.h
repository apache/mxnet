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
                    in_attrs->at(0).Size() / 16 + 3:
                    in_attrs->at(0).Size() / 16 + 4;
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

struct init_threshold_2bit {
  MSHADOW_XINLINE static void Map(int i,
                                  float *out,
                                  const float neg_threshold,
                                  const float pos_threshold,
                                  int size) {
    // The first two elements in output are thresholds
    // The third element is the original size of the array
    out[0] = neg_threshold;
    out[1] = pos_threshold;
    out[2] = (float)size;
  }
};

struct quantize_2bit {
  MSHADOW_XINLINE static void Map(int block_id,
                                  int gradsize,
                                  float *out,
                                  float *grad,
                                  float *residual,
                                  const float neg_threshold,
                                  const float pos_threshold) {

//    int num = 1;
//    if(*(char *)&num == 1)
//    {
//      std::cout<<"Little-Endian"<<std::endl;
//    }
//    else
//    {
//      std::cout<<"Big-Endian"<<std::endl;
//    }

    float* out_block = out + block_id;
    // start and end are indices in original grad array
    int start = block_id*16;
    int end = ( start + 16 <= gradsize) ? start+16 : gradsize;
    char* ch_ptr = reinterpret_cast<char*>(out_block);
    for (int i=start; i<end; i++){
      grad[i] += residual[i];
      char* curr_ptr = ch_ptr + (i-start)/4;
      if (grad[i] >= pos_threshold) {
        residual[i] = grad[i] - pos_threshold;
        // set data to 10
//        std::cout<<"or "<<(2u<<(6-((i%4)*2)))<<std::endl;
        (*curr_ptr) |= (2u<<(6-((i%4)*2)));
      } else if (grad[i] <= neg_threshold) {
        residual[i] = grad[i] - neg_threshold;
        // set data to 01
//        std::cout<<"or "<<(1u<<(6-((i%4)*2)))<<std::endl;
        (*curr_ptr) |= (1u<<(6-((i%4)*2)));
      } else {
        // leave data as 00
        residual[i] = grad[i];
      }
//      std::cout<<grad[i]<<std::endl;
    }

//    std::cout<<*out_block<<std::endl;
    std::string fstr;
    union { float f; uint32_t i; } u;
    u.f = *out_block;
    fstr.clear();

    for (int i = 0; i < 32; i++)
    {
      if (u.i % 2)  fstr.push_back('1');
      else fstr.push_back('0');
      u.i >>= 1;
    }

    // Reverse the string since now it's backwards
    std::string temp(fstr.rbegin(), fstr.rend());
    fstr = temp;

//    floatToBinary3(*out_block, fstr);
//    std::cout<<fstr<<std::endl;
  }
};

template<typename xpu>
void Quantize2BitImpl(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs,
                      const float neg_threshold, const float pos_threshold) {
  // First, init the memory of output to 0x00000000
  mxnet_op::Kernel<init_mem_2bit, xpu>::Launch(s, inputs[2].Size(),
                              inputs[2].dptr<float>());  // compressed array
  // Then, init threshold and original size
  mxnet_op::Kernel<init_threshold_2bit, xpu>::Launch(s, 1,
                              inputs[2].dptr<float>(),   // compressed array
                              neg_threshold, pos_threshold,
                              inputs[0].Size());
  // Finally, compress the data and calculate new residual
  mxnet_op::Kernel<quantize_2bit, xpu>::Launch(s, inputs[2].Size()-3,
                          inputs[0].Size(),
                          inputs[2].dptr<float>()+3,   // compressed array
                          inputs[0].dptr<float>(),     // input array
                          inputs[1].dptr<float>(),     // residual array
                          neg_threshold,     // negative threshold
                          pos_threshold);    // positive threshold
}

// function defined as operator
template<typename xpu>
void Quantize2BitCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TwoBitParam& param = nnvm::get<TwoBitParam>(attrs.parsed);
  Quantize2BitImpl<xpu>(s, inputs, param.neg_threshold, param.pos_threshold);
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
                    in_attrs->at(0).Size() / 16 + 3:
                    in_attrs->at(0).Size() / 16 + 4;
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
                                  float *neg_threshold,
                                  float *pos_threshold) {
    // get block ptr
    int block_id = i / 16;
    char* ch_ptr = reinterpret_cast<char*>(in+block_id);
    // get row ptr
    int row_id = (i%16)/4;
    ch_ptr += row_id;
    // get column id
    int col_id = (i%16)%4;
    // Decompress
    switch (col_id) {
      case 0:
        // positve
        if (((*ch_ptr) & (0xc0)) == 0x80) {  // binary: (10)00 0000
          out[i] = *pos_threshold;
        // negative
        } else if (((*ch_ptr) & (0xc0)) == 0x40) {  // binary: (01)00 0000
          out[i] = *neg_threshold;
        } else {  // 0
          out[i] = 0;
        }
        break;
      case 1:
        // positve
        if (((*ch_ptr) & (0x30)) == 0x20) {  // binary: 00(10) 0000
          out[i] = *pos_threshold;
        // negative
        } else if (((*ch_ptr) & (0x30)) == 0x10) {  // binary: 00(01) 0000
          out[i] = *neg_threshold;
        } else {  // 0
          out[i] = 0;
        }
        break;
      case 2:
        // positve
        if (((*ch_ptr) & (0x0c)) == 0x08) {  // binary: 00(10) 0000
          out[i] = *pos_threshold;
        // negative
        } else if (((*ch_ptr) & (0x0c)) == 0x04) {  // binary: 00(01) 0000
          out[i] = *neg_threshold;
        } else {  // 0
          out[i] = 0;
        }
        break;
      case 3:
        // positve
        if (((*ch_ptr) & (0x03)) == 0x02) {  // binary: 00(10) 0000
          out[i] = *pos_threshold;
        // negative
        } else if (((*ch_ptr) & (0x03)) == 0x01) {  // binary: 00(01) 0000
          out[i] = *neg_threshold;
        } else {  // 0
          out[i] = 0;
        }
        break;
      default:
        break;
    }
  }
};

template<typename xpu>
void Dequantize2BitImpl(mshadow::Stream<xpu>* s, const std::vector<TBlob>& inputs) {
  // Can only decompress the float32 data
  mxnet_op::Kernel<dequantize_2bit, xpu>::Launch(s, inputs[1].Size(),  // original size
                              inputs[1].dptr<float>(),        // out array
                              inputs[0].dptr<float>()+3,      // compressed array
                              inputs[0].dptr<float>(),     // negative threshold
                              inputs[0].dptr<float>()+1);  // positive threshold
}

template<typename xpu>
void Dequantize2BitCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  Dequantize2BitImpl<xpu>(s, inputs);
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
  //TODO(huilgolr) check
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
