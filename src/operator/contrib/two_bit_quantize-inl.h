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
    *reinterpret_cast<int*>(out+i) = 0x00000000;
  }
};

struct init_threshold_2bit {
  MSHADOW_XINLINE static void Map(int i,
                                  float *out,
                                  const float *neg_threshold,
                                  const float *pos_threshold) {
    // The first two elments in output is threshold
    out[0] = *neg_threshold;
    out[1] = *pos_threshold;
  }
};

struct quantize_2bit {
  MSHADOW_XINLINE static void Map(int i,
                                  float *out,
                                  float *grad,
                                  float *residual,
                                  const float *neg_threshold,
                                  const float *pos_threshold) {
    // Add residual to gradient
    grad[i] += residual[i];
    // get block id
    int block_id = i / 16;
    char* ch_ptr = reinterpret_cast<char*>(out+block_id);
    // get row ptr
    int row_id = (i%16)/4;
    ch_ptr += row_id;
    // get column id
    int col_id = (i%16)%4;
    // Compress
    if (grad[i] <= *neg_threshold) {  // set data to 01
      // new residual
      residual[i] = grad[i] - *neg_threshold;
      switch (col_id) {
        case 0:
          (*ch_ptr) |= 0x40;  // binary: (01)00 0000
          break;
        case 1:
          (*ch_ptr) |= 0x10;  // binary: 00(01) 0000
          break;
        case 2:
          (*ch_ptr) |= 0x04;  // binary: 0000 (01)00
          break;
        case 3:
          (*ch_ptr) |= 0x01;  // binary: 0000 00(01)
          break;
        default:
          break;
      }
    } else if (grad[i] >= *pos_threshold) {  // set data to 10
      residual[i] = grad[i] - *pos_threshold;
      switch (col_id) {
        case 0:
          (*ch_ptr) |= 0x80;  // binary: (10)00 0000
          break;
        case 1:
          (*ch_ptr) |= 0x20;  // binary: 00(10) 0000
          break;
        case 2:
          (*ch_ptr) |= 0x08;  // binary: 0000 (10)00
          break;
        case 3:
          (*ch_ptr) |= 0x02;  // binary: 0000 00(10)
          break;
        default:
          break;
      }
    } else {  // else 00
      residual[i] = grad[i];
    }
  }
};

template<typename xpu>
void Quantize2BitCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  // For now, this method can only compress the float data
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // First, init the memory of output to 0x00000000
  Kernel<init_mem_2bit, xpu>::Launch(s, outputs[0].Size(),
                              outputs[0].dptr<float>());  // output array
  // Then, init threshold
  Kernel<init_threshold_2bit, xpu>::Launch(s, 1,
                              outputs[0].dptr<float>(),  // output array
                              inputs[2].dptr<float>(),   // negative threshold
                              inputs[3].dptr<float>());  // positive threshold
  // Finally, compress the data and calculate new residual
  Kernel<quantize_2bit, xpu>::Launch(s, inputs[0].Size(),
                          outputs[0].dptr<float>()+2,  // output array
                          inputs[0].dptr<float>(),     // input array
                          inputs[1].dptr<float>(),     // residual array
                          inputs[2].dptr<float>(),     // negative threshold
                          inputs[3].dptr<float>());    // positive threshold
}

inline bool Quantize2BitShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  // 0. input array
  // 1. residual array
  // 2. negative threshold
  // 3. positive threshold
  CHECK_EQ(in_attrs->size(), 4U);
  // 0. output array
  CHECK_EQ(out_attrs->size(), 1U);
  // check input
  CHECK(!shape_is_none(in_attrs->at(0)));
  CHECK(!shape_is_none(in_attrs->at(1)));
  CHECK(shape_is_scalar(in_attrs->at(2)));
  CHECK(shape_is_scalar(in_attrs->at(3)));
  CHECK_EQ(in_attrs->at(0).Size(),
           in_attrs->at(1).Size());
  // check output
  int shape = in_attrs->at(0).Size() % 16 == 0 ?
                    in_attrs->at(0).Size() / 16 + 2:
                    in_attrs->at(0).Size() / 16 + 3;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape{shape});
  // new residual array will re-use the memory of
  // the original residual array
  return true;
}

inline bool Quantize2BitType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  // 0. input array
  // 1. residual array
  // 2. negative threshold
  // 3. positive threshold
  CHECK_EQ(in_attrs->size(), 4U);
  // 0. output array
  CHECK_EQ(out_attrs->size(), 1U);
  // check input
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`quantize_2bit_` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "`quantize_2bit_` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32)
    << "the third input of `quantize_2bit` should be "
    << "a tensor with type of float";
  CHECK_EQ((*in_attrs)[3], mshadow::kFloat32)
    << "the fourth input of `quantize_2bit` should be "
    << "a tensor with type of float";
  // check output
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  // new residual array will re-use the memory of
  // the original residual array
  return true;
}

struct dequantize_2bit {
  // Decompress
  MSHADOW_XINLINE static void Map(int i,
                                  float *out,
                                  float *in,
                                  const float *neg_threshold,
                                  const float *pos_threshold) {
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
void Dequantize2BitCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // For now, this method can only decompress the float data
  Kernel<dequantize_2bit, xpu>::Launch(s, inputs[1].Size(),  // original size
                              inputs[1].dptr<float>(),       // out array
                              inputs[0].dptr<float>()+2,     // compressed array
                              inputs[0].dptr<float>(),     // negative threshold
                              inputs[0].dptr<float>()+1);  // positve threshold
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
  // No output
  CHECK_EQ(out_attrs->size(), 0U);
  // check input
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`dquantize_2bit_` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "`dquantize_2bit_` only supports float32 input for now";
  return true;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_TWO_BIT_QUANTIZE_INL_H_
