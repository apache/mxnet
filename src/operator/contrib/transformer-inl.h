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
 *  Copyright (c) 2018 by Contributors
 * \file transformer-inl.h
 * \brief Function used in cc and cu
 */
#ifndef MXNET_OPERATOR_CONTRIB_TRANSFORMER_INL_H_
#define MXNET_OPERATOR_CONTRIB_TRANSFORMER_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../mshadow_op.h"


namespace mxnet {
namespace op {

struct InterleavedMatMulParam : public dmlc::Parameter<InterleavedMatMulParam> {
  int heads;
  bool bwd_ignore_zero_init;
  DMLC_DECLARE_PARAMETER(InterleavedMatMulParam) {
    DMLC_DECLARE_FIELD(heads)
    .describe("Set number of heads");
  }
};

template<typename xpu>
static void DivSqrtDimForward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_GE(inputs[0].ndim(), 1);
  int last_idx = inputs[0].ndim() - 1;
  double sqrt_dim = std::sqrt(static_cast<double>(inputs[0].shape_[last_idx]));
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::div, Req>, xpu>::Launch(
        s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), DType(sqrt_dim));
    });
  });
}



struct SldWinAttenParam : public dmlc::Parameter<SldWinAttenParam> {
  int w;
  bool symmetric;
  DMLC_DECLARE_PARAMETER(SldWinAttenParam) {
    DMLC_DECLARE_FIELD(w)
    .describe("The one-sided window length");
    DMLC_DECLARE_FIELD(symmetric)
    .describe("If false, each token will only attend to itself and the previous tokens.");
  }
};


struct SldWinAttenMaskLike {
  MSHADOW_XINLINE static void Map(int i, float *out, int32_t *dilation, int32_t *val_length,
                                  bool symmetric, int w, int seq_length, int num_heads) {
    out[i] = 1;
    int w_len = symmetric ? (w + w + 1) : (w + 1);
    int idx_0 = i / (seq_length * num_heads * w_len);  // batch idx
    int tmp = i % (seq_length * num_heads * w_len);
    int idx_1 = tmp / (num_heads * w_len);  // sequence idx
    tmp = tmp % (num_heads * w_len);
    int idx_2 = tmp / w_len;  // head idx
    int idx_3 = tmp % w_len;  // win idx

    bool is_zero = idx_3 < (w - idx_1/dilation[idx_2]) || idx_1 >= val_length[idx_0] \
      || (symmetric && (w_len-idx_3-1) < (w - (val_length[idx_0]-idx_1-1)/dilation[idx_2]));
    if (is_zero) out[i] = 0;
  }
};


template<typename xpu>
void SldWinAttenMaskLikeForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "Currently only support kWriteTo";
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const SldWinAttenParam& param = nnvm::get<SldWinAttenParam>(attrs.parsed);
  CHECK_EQ(outputs[0].type_flag_, kFloat32);
  float* out = outputs[0].dptr<float>();
  CHECK_EQ(inputs[1].type_flag_, kInt32);
  int32_t* dilation = inputs[1].dptr<int32_t>();
  CHECK_EQ(inputs[2].type_flag_, kInt32);
  int32_t* val_length = inputs[2].dptr<int32_t>();

  int seq_length = inputs[0].shape_[1];
  int num_heads = inputs[0].shape_[2];
  int num_threads = outputs[0].Size();

  mxnet_op::Kernel<SldWinAttenMaskLike, xpu>::Launch(s, num_threads, out, dilation,
      val_length, param.symmetric, param.w, seq_length, num_heads);
}




struct DiagMM {
  MSHADOW_XINLINE static void Map(int tid, float *out, float *lhs, float *rhs,
                                  int32_t *dilation, int batch_size, int seq_length,
                                  int num_heads, int out_last_dim, int lhs_last_dim, int w,
                                  int w_right, bool diagonal_lhs, bool transpose_lhs) {
    out[tid] = 0;
    int stride = seq_length * num_heads * out_last_dim;
    int idx_0 = tid / stride;  // batch idx
    int tmp = tid % stride;
    stride = num_heads * out_last_dim;
    int idx_1 = tmp / stride;  // sequence idx
    tmp = tmp % stride;
    int idx_2 = tmp / out_last_dim;  // head idx
    int idx_3 = tmp % out_last_dim;  // window idx or hidden feature idx

    if (!diagonal_lhs) {
      int tmp_idx = idx_1 + dilation[idx_2] * (idx_3 - w);
      if (tmp_idx >= seq_length || tmp_idx < 0) return;
      for (int i = 0; i < lhs_last_dim; i++) {
        int lhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
          idx_1 * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + i;
        int rhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
          tmp_idx * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + i;
        out[tid] += lhs[lhs_idx] * rhs[rhs_idx];
      }
    } else {
      if (!transpose_lhs) {
        for (int i = 0; i < lhs_last_dim; i++) {
          int tmp_idx = idx_1 + dilation[idx_2] * (i - w);
          if (tmp_idx >= seq_length || tmp_idx < 0) continue;
          int lhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
            idx_1 * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + i;
          int rhs_idx = idx_0 * (seq_length * num_heads * out_last_dim) + \
            tmp_idx * (num_heads * out_last_dim) + idx_2 * out_last_dim + idx_3;
          out[tid] += lhs[lhs_idx] * rhs[rhs_idx];
        }
      } else {
        for (int i = 0; i < lhs_last_dim; i++) {
          int tmp_idx = idx_1 + dilation[idx_2] * (i - w_right);
          if (tmp_idx >= seq_length || tmp_idx < 0) continue;
          int lhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
            tmp_idx * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + ((w_right + w) - i);
          int rhs_idx = idx_0 * (seq_length * num_heads * out_last_dim) + \
            tmp_idx * (num_heads * out_last_dim) + idx_2 * out_last_dim + idx_3;
          out[tid] += lhs[lhs_idx] * rhs[rhs_idx];
        }
      }
    }
  }
};



template<typename xpu>
void DiagMMImpl(const OpContext& ctx, const TBlob& out, const TBlob& lhs,
                const TBlob& rhs, const TBlob& dilation, bool diagonal_lhs,
                bool transpose_lhs, int w, int w_right) {
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  CHECK_EQ(out.type_flag_, kFloat32);
  CHECK_EQ(lhs.type_flag_, kFloat32);
  CHECK_EQ(rhs.type_flag_, kFloat32);
  CHECK_EQ(dilation.type_flag_, kInt32);

  float* lhs_data = lhs.dptr<float>();
  float* rhs_data = rhs.dptr<float>();
  int32_t* dilation_data = dilation.dptr<int32_t>();
  float* out_data = out.dptr<float>();

  int batch_size = lhs.shape_[0];
  int seq_length = lhs.shape_[1];
  int num_heads = lhs.shape_[2];
  int lhs_last_dim = lhs.shape_[3];
  int out_last_dim = out.shape_[3];
  int num_threads = out.Size();

  mxnet_op::Kernel<DiagMM, xpu>::Launch(s, num_threads, out_data, lhs_data, rhs_data,
      dilation_data, batch_size, seq_length, num_heads, out_last_dim, lhs_last_dim, w,
      w_right, diagonal_lhs, transpose_lhs);
}


template<typename xpu>
void SldWinAttenScoreForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  using namespace mshadow;
  const SldWinAttenParam& param = nnvm::get<SldWinAttenParam>(attrs.parsed);
  int w_right = param.symmetric ? param.w : 0;
  DiagMMImpl<xpu>(ctx, outputs.at(0), inputs.at(0), inputs.at(1), inputs.at(2),
    false, false, param.w, w_right);
}


template<typename xpu>
void SldWinAttenScoreBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 3U);
  using namespace mshadow;
  const SldWinAttenParam& param = nnvm::get<SldWinAttenParam>(attrs.parsed);
  int w_right = param.symmetric ? param.w : 0;
  // grad_query = matmul(grad_score, key)
  DiagMMImpl<xpu>(ctx, outputs.at(0), inputs.at(0), inputs.at(2), inputs.at(3),
      true, false, param.w, w_right);
  // grad_key = matmul(grad_score.T, query)
  DiagMMImpl<xpu>(ctx, outputs.at(1), inputs.at(0), inputs.at(1), inputs.at(3),
      true, true, param.w, w_right);
}



template<typename xpu>
void SldWinAttenContextForward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  using namespace mshadow;
  const SldWinAttenParam& param = nnvm::get<SldWinAttenParam>(attrs.parsed);
  int w_right = param.symmetric ? param.w : 0;
  // context_vec = matmul(score, value)
  DiagMMImpl<xpu>(ctx, outputs.at(0), inputs.at(0), inputs.at(1), inputs.at(2),
      true, false, param.w, w_right);
}


template<typename xpu>
void SldWinAttenContextBackward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 3U);
  using namespace mshadow;
  const SldWinAttenParam& param = nnvm::get<SldWinAttenParam>(attrs.parsed);
  int w_right = param.symmetric ? param.w : 0;
  // grad_score = matmul(grad_context, value.T)
  DiagMMImpl<xpu>(ctx, outputs.at(0), inputs.at(0), inputs.at(2), inputs.at(3),
      false, false, param.w, w_right);
  // grad_value = matmul(score.T, grad_context)
  DiagMMImpl<xpu>(ctx, outputs.at(1), inputs.at(1), inputs.at(0), inputs.at(3),
      true, true, param.w, w_right);
}




}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_TRANSFORMER_INL_H_
