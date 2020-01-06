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
 * \file transformer.cc
 * \brief CPU implementation of the operators used in Transformer
 */
#include <mxnet/base.h>
#include "./transformer-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InterleavedMatMulParam);

static bool InterleavedMatMulSelfAttQKShape(const NodeAttrs& attrs,
                                            mxnet::ShapeVector* in_shape,
                                            mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U) << "Input:[queries_keys_values] currently have, "
                                 << in_shape->size() << " inputs";
  auto qkv_shape = in_shape->at(0);
  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  out_shape->resize(1);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({params.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));
  return true;
}

static bool InterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  CHECK_EQ(in_shape->size(), 2U) << "Input:[queries_keys_values, attention] currently have, "
                                 << in_shape->size() << " inputs";
  auto qkv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-3*proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  CHECK_EQ(att_shape.ndim(), 3U)
    << "Input attention should be 3D in batch-seq_length-seq_length, "
    << "currently is: " << att_shape.ndim() << "D";
  CHECK_EQ(qkv_shape[0], att_shape[1])
    << "queries_keys_values.shape[0] and attention.shape[1] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[1];
  CHECK_EQ(qkv_shape[0], att_shape[2])
    << "queries_keys_values.shape[0] and attention.shape[2] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[2];
  CHECK_EQ(qkv_shape[2] % 3, 0)
    << "queries_keys_values.shape[2] should be a multiple of 3, "
    << "currently is " << qkv_shape[2];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
  return true;
}

static bool InterleavedMatMulEncDecQKShape(const NodeAttrs& attrs,
                                           mxnet::ShapeVector* in_shape,
                                           mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input:[queries, keys_values], currently have "
                                 << in_shape->size() << " inputs";
  auto q_shape = in_shape->at(0);
  auto kv_shape = in_shape->at(1);
  CHECK_EQ(q_shape.ndim(), 3U) << "Input queries should be 3D in seq_length-batch-proj_dim, "
                               << "currently is " << q_shape.ndim() << "D";
  CHECK_EQ(kv_shape.ndim(), 3U) << "Input queries should be 3D in seq_length-batch-2*proj_dim, "
                                << "currently is " << kv_shape.ndim() << "D";
  CHECK_EQ(q_shape[2] * 2, kv_shape[2])
    << "keys_values.shape[2] should be equal to queries.shape[2] * 2, "
    << "currently are: " << kv_shape[2] << " and " << q_shape[2];
  CHECK_EQ(q_shape[1], kv_shape[1])
    << "queries.shape[1] should be equal to keys_values.shape[1], "
    << "currently are: " << q_shape[1] << " and " << kv_shape[1];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({q_shape[1] * params.heads, q_shape[0], kv_shape[0]}));
  return true;
}

static bool InterleavedMatMulEncDecValAttShape(const NodeAttrs& attrs,
                                               mxnet::ShapeVector* in_shape,
                                               mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input: [keys_values, attention], currently have "
                                 << in_shape->size() << " inputs";
  auto kv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(kv_shape.ndim(), 3U)
    << "Input keys_values should be 3D in seq_length-batch-2*proj_dim, "
    << "currently is " << kv_shape.ndim() << "D";
  CHECK_EQ(att_shape.ndim(), 3U)
    << "Input attention should be 3D in batch-seq_length-seq_length, "
    << "currently is " << att_shape.ndim() << "D";
  CHECK_EQ(kv_shape[0], att_shape[2])
    << "keys_values.shape[0] should be equal to attention.shape[2], currently are "
    << kv_shape[0] << " and " << att_shape[2];
  CHECK_EQ(kv_shape[1] * params.heads, att_shape[0]) << "attention.shape[0] "
    << "should be equal to keys_values.shape[1] * heads, currently are: "
    << att_shape[2] << " and " << kv_shape[1];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({att_shape[1], kv_shape[1], kv_shape[2] / 2}));
  return true;
}

void strided_batch_sgemm(bool transA, bool transB,
                         index_t m, index_t n, index_t k,
                         float alpha, const float *a, index_t lda,
                         index_t strideA, const float *b, index_t ldb,
                         index_t strideB, float beta, float *c, index_t ldc,
                         index_t strideC, int32_t batchCount) {
  std::vector<const float*> pp_A(batchCount, nullptr);
  std::vector<const float*> pp_B(batchCount, nullptr);
  std::vector<float*> pp_C(batchCount, nullptr);

  for (int i = 0; i < batchCount; i++) {
    pp_A[i] = a + i * strideA;
    pp_B[i] = b + i * strideB;
    pp_C[i] = c + i * strideC;
  }

#if (MSHADOW_USE_MKL && INTEL_MKL_VERSION >= 20160000)
  const int GROUP_SIZE = 1;
  MKL_INT p_m[GROUP_SIZE] = {m};
  MKL_INT p_n[GROUP_SIZE] = {n};
  MKL_INT p_k[GROUP_SIZE] = {k};
  MKL_INT p_lda[GROUP_SIZE] = {lda};
  MKL_INT p_ldb[GROUP_SIZE] = {ldb};
  MKL_INT p_ldc[GROUP_SIZE] = {ldc};

  float p_alpha[GROUP_SIZE] = {alpha};
  float p_beta[GROUP_SIZE] = {beta};

  CBLAS_TRANSPOSE cblas_a_trans = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE cblas_b_trans = transB ? CblasTrans : CblasNoTrans;

  MKL_INT p_group_sizeb[GROUP_SIZE] = {batchCount};
  CBLAS_TRANSPOSE p_transa[GROUP_SIZE] = {cblas_a_trans};
  CBLAS_TRANSPOSE p_transb[GROUP_SIZE] = {cblas_b_trans};

  cblas_sgemm_batch(CblasColMajor, p_transa, p_transb,
                    p_m, p_n, p_k, p_alpha, pp_A.data(), p_lda, pp_B.data(),
                    p_ldb, p_beta, pp_C.data(), p_ldc, GROUP_SIZE, p_group_sizeb);
#else
  for (int i = 0; i < batchCount; ++i) {
    cblas_sgemm(CblasColMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k,
                alpha, pp_A[i], lda,
                pp_B[i], ldb, beta, pp_C[i], ldc);
  }
#endif
}

void InterleavedMatMulSelfAttQKCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);

  if (req[0] == kNullOp)
    return;

  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  float* output = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

  const index_t qkv_seq_len    = inputs[0].shape_[0];
  const index_t sequences      = inputs[0].shape_[1];
  const index_t output_lin_dim = inputs[0].shape_[2];
  const index_t embed_dim      = output_lin_dim / 3;
  const index_t head_dim       = embed_dim / params.heads;
  const index_t attn_batches   = params.heads * sequences;
  const index_t lead_dim       = attn_batches * 3 * head_dim;
  const index_t batch_stride   = 3 * head_dim;
  const float beta             = req[0] == kAddTo ? 1.f : 0.f;
  const float scale            = 1.0 / sqrt(static_cast<float>(head_dim));

  strided_batch_sgemm(true,
                      false,
                      qkv_seq_len,
                      qkv_seq_len,
                      head_dim,
                      scale,
                      queries_keys_values + head_dim,
                      lead_dim,
                      batch_stride,
                      queries_keys_values,
                      lead_dim,
                      batch_stride,
                      beta,
                      output,
                      qkv_seq_len,
                      qkv_seq_len * qkv_seq_len,
                      attn_batches);
}

void BackwardInterleavedMatMulSelfAttQKCPU(const nnvm::NodeAttrs& attrs,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  if (req[0] == kNullOp)
    return;

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  const float* output_grads        = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* queries_keys_values = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  float* queries_keys_values_grads = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const index_t qkv_seq_len    = inputs[1].shape_[0];
  const index_t sequences      = inputs[1].shape_[1];
  const index_t output_lin_dim = inputs[1].shape_[2];
  const index_t embed_dim      = output_lin_dim / 3;
  const index_t head_dim       = embed_dim / params.heads;
  const index_t attn_batches   = params.heads * sequences;
  const index_t lead_dim       = attn_batches * 3 * head_dim;
  const index_t batch_stride   = 3 * head_dim;
  const float scale            = 1.0 / sqrt(static_cast<float>(head_dim));
  const float beta = req[0] == kAddTo ? 1.f : 0.f;

  if (req[0] == kWriteTo) {
    memset(queries_keys_values_grads, 0, outputs[0].shape_.Size() * sizeof (float));
  }

  strided_batch_sgemm(false,
                      false,
                      head_dim,
                      qkv_seq_len,
                      qkv_seq_len,
                      scale,
                      queries_keys_values + head_dim,
                      lead_dim,
                      batch_stride,
                      output_grads,
                      qkv_seq_len,
                      qkv_seq_len * qkv_seq_len,
                      beta,
                      queries_keys_values_grads,
                      lead_dim,
                      batch_stride,
                      attn_batches);

  strided_batch_sgemm(false,
                      true,
                      head_dim,
                      qkv_seq_len,
                      qkv_seq_len,
                      scale,
                      queries_keys_values,
                      lead_dim,
                      batch_stride,
                      output_grads,
                      qkv_seq_len,
                      qkv_seq_len * qkv_seq_len,
                      beta,
                      queries_keys_values_grads + head_dim,
                      lead_dim,
                      batch_stride,
                      attn_batches);
}

void InterleavedMatMulSelfAttValAttCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  if (req[0] == kNullOp)
    return;

  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* attention_maps      = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  float* output                    = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const index_t qkv_seq_len    = inputs[0].shape_[0];
  const index_t sequences      = inputs[0].shape_[1];
  const index_t output_lin_dim = inputs[0].shape_[2];
  const index_t embed_dim      = output_lin_dim / 3;
  const index_t head_dim       = embed_dim / params.heads;
  const index_t attn_batches   = params.heads * sequences;
  const index_t lead_dim       = attn_batches * 3 * head_dim;
  const index_t batch_stride   = 3 * head_dim;
  const float alpha             = 1.f;
  const float beta              = req[0] == kAddTo ? 1.f : 0.f;

  strided_batch_sgemm(false,
                      false,
                      head_dim,
                      qkv_seq_len,
                      qkv_seq_len,
                      alpha,
                      queries_keys_values + 2 * head_dim,
                      lead_dim,
                      batch_stride,
                      attention_maps,
                      qkv_seq_len,
                      qkv_seq_len * qkv_seq_len,
                      beta,
                      output,
                      head_dim * attn_batches,
                      head_dim,
                      attn_batches);
}

void BackwardInterleavedMatMulSelfAttValAttCPU(const nnvm::NodeAttrs& attrs,
                                               const OpContext &ctx,
                                               const std::vector<TBlob> &inputs,
                                               const std::vector<OpReqType> &req,
                                               const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  if (req[0] == kNullOp)
    return;

  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* output_grads              = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* queries_keys_values       = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  const float* attention_maps            = inputs[2].FlatTo2D<cpu, float>(s).dptr_;
  float* queries_keys_values_grads       = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  float* attention_maps_grads            = outputs[1].FlatTo2D<cpu, float>(s).dptr_;
  const index_t qkv_seq_len    = inputs[1].shape_[0];
  const index_t sequences      = inputs[1].shape_[1];
  const index_t output_lin_dim = inputs[1].shape_[2];
  const index_t embed_dim      = output_lin_dim / 3;
  const index_t head_dim       = embed_dim / params.heads;
  const index_t attn_batches   = params.heads * sequences;
  const index_t lead_dim       = attn_batches * 3 * head_dim;
  const index_t batch_stride   = 3 * head_dim;
  const float alpha            = 1.f;
  if (req[0] != kNullOp) {
    if (req[0] == kWriteTo) {
      memset(queries_keys_values_grads, 0, outputs[0].shape_.Size() * sizeof (float));
    }

    const float beta = req[0] == kAddTo ? 1.f : 0.f;
    strided_batch_sgemm(false,
                        true,
                        head_dim,
                        qkv_seq_len,
                        qkv_seq_len,
                        alpha,
                        output_grads,
                        head_dim * attn_batches,
                        head_dim,
                        attention_maps,
                        qkv_seq_len,
                        qkv_seq_len * qkv_seq_len,
                        beta,
                        queries_keys_values_grads + 2 * head_dim,
                        lead_dim,
                        batch_stride,
                        attn_batches);
  }
  if (req[1] != kNullOp) {
    const float beta = req[1] == kAddTo ? 1.f : 0.f;
    strided_batch_sgemm(true,
                        false,
                        qkv_seq_len,
                        qkv_seq_len,
                        head_dim,
                        alpha,
                        queries_keys_values + 2 * head_dim,
                        lead_dim,
                        batch_stride,
                        output_grads,
                        head_dim * attn_batches,
                        head_dim,
                        beta,
                        attention_maps_grads,
                        qkv_seq_len,
                        qkv_seq_len * qkv_seq_len,
                        attn_batches);
  }
}

void InterleavedMatMulEncDecQKCPU(const nnvm::NodeAttrs& attrs,
                                  const OpContext &ctx,
                                  const std::vector<TBlob> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  if (req[0] == kNullOp)
    return;

  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* queries     = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* keys_values = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  float* output            = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const index_t q_seq_len         = inputs[0].shape_[0];
  const index_t sequences         = inputs[0].shape_[1];
  const index_t output_lin_q_dim  = inputs[0].shape_[2];
  const index_t kv_seq_len        = inputs[1].shape_[0];
  const index_t embed_dim         = output_lin_q_dim;
  const index_t head_dim          = embed_dim / params.heads;
  const index_t attn_batches      = params.heads * sequences;
  const index_t lead_dim_q        = attn_batches * head_dim;
  const index_t lead_dim_kv       = attn_batches * 2 * head_dim;
  const index_t batch_stride_q    = head_dim;
  const index_t batch_stride_kv   = head_dim * 2;
  const float beta                = req[0] == kAddTo ? 1.f : 0.f;
  const float scale               = 1.f / sqrt(static_cast<float>(head_dim));

  strided_batch_sgemm(true,
                      false,
                      kv_seq_len,
                      q_seq_len,
                      head_dim,
                      scale,
                      keys_values,
                      lead_dim_kv,
                      batch_stride_kv,
                      queries,
                      lead_dim_q,
                      batch_stride_q,
                      beta,
                      output,
                      kv_seq_len,
                      kv_seq_len * q_seq_len,
                      attn_batches);
}

void BackwardInterleavedMatMulEncDecQKCPU(const nnvm::NodeAttrs& attrs,
                                          const OpContext &ctx,
                                          const std::vector<TBlob> &inputs,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  if (req[0] == kNullOp)
    return;

  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* output_grads = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* queries       = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  const float* keys_values   = inputs[2].FlatTo2D<cpu, float>(s).dptr_;
  float* queries_grads       = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  float* keys_values_grads   = outputs[1].FlatTo2D<cpu, float>(s).dptr_;
  const index_t q_seq_len         = inputs[1].shape_[0];
  const index_t sequences         = inputs[1].shape_[1];
  const index_t output_lin_q_dim  = inputs[1].shape_[2];
  const index_t kv_seq_len        = inputs[2].shape_[0];
  const index_t embed_dim         = output_lin_q_dim;
  const index_t head_dim          = embed_dim / params.heads;
  const index_t attn_batches      = params.heads * sequences;
  const index_t lead_dim_q        = attn_batches * head_dim;
  const index_t lead_dim_kv       = attn_batches * 2 * head_dim;
  const index_t batch_stride_q    = head_dim;
  const index_t batch_stride_kv   = head_dim * 2;
  const float scale               = 1.f / sqrt(static_cast<float>(head_dim));

  if (req[0] != kNullOp) {
    const float beta = req[0] == kAddTo ? 1.f : 0.f;
    strided_batch_sgemm(false,
                        false,
                        head_dim,
                        q_seq_len,
                        kv_seq_len,
                        scale,
                        keys_values,
                        lead_dim_kv,
                        batch_stride_kv,
                        output_grads,
                        kv_seq_len,
                        kv_seq_len * q_seq_len,
                        beta,
                        queries_grads,
                        lead_dim_q,
                        batch_stride_q,
                        attn_batches);
  }
  if (req[1] != kNullOp) {
    if (req[1] == kWriteTo) {
      memset(keys_values_grads, 0, outputs[1].shape_.Size() * sizeof (float));
    }
    const float beta = req[1] == kAddTo ? 1.f : 0.f;
    strided_batch_sgemm(false,
                        true,
                        head_dim,
                        kv_seq_len,
                        q_seq_len,
                        scale,
                        queries,
                        lead_dim_q,
                        batch_stride_q,
                        output_grads,
                        kv_seq_len,
                        kv_seq_len * q_seq_len,
                        beta,
                        keys_values_grads,
                        lead_dim_kv,
                        batch_stride_kv,
                        attn_batches);
  }
}

void InterleavedMatMulEncDecValAttCPU(const nnvm::NodeAttrs& attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  if (req[0] == kNullOp)
    return;

  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* keys_values    = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* attention_maps = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  float* output               = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const index_t kv_seq_len        = inputs[0].shape_[0];
  const index_t output_lin_kv_dim = inputs[0].shape_[2];
  const index_t attn_batches      = inputs[1].shape_[0];
  const index_t q_seq_len         = inputs[1].shape_[1];
  const index_t embed_dim         = output_lin_kv_dim / 2;
  const index_t head_dim          = embed_dim / params.heads;
  const index_t lead_dim_kv       = attn_batches * head_dim * 2;
  const index_t batch_stride_kv   = 2 * head_dim;
  const float alpha               = 1.f;
  const float beta                = req[0] == kAddTo ? 1.f : 0.f;

  strided_batch_sgemm(false,
                      false,
                      head_dim,
                      q_seq_len,
                      kv_seq_len,
                      alpha,
                      keys_values + head_dim,
                      lead_dim_kv,
                      batch_stride_kv,
                      attention_maps,
                      kv_seq_len,
                      kv_seq_len * q_seq_len,
                      beta,
                      output,
                      head_dim * attn_batches,
                      head_dim,
                      attn_batches);
}

void BackwardInterleavedMatMulEncDecValAttCPU(const nnvm::NodeAttrs& attrs,
                                              const OpContext &ctx,
                                              const std::vector<TBlob> &inputs,
                                              const std::vector<OpReqType> &req,
                                              const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, mshadow::kFloat32)
    << "Only FP32 is supported on CPU at the moment";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const float* output_grads   = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
  const float* keys_values    = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
  const float* attention_maps = inputs[2].FlatTo2D<cpu, float>(s).dptr_;
  float* keys_values_grads    = outputs[0].FlatTo2D<cpu, float>(s).dptr_;
  float* attention_maps_grads = outputs[1].FlatTo2D<cpu, float>(s).dptr_;
  const index_t kv_seq_len        = inputs[1].shape_[0];
  const index_t output_lin_kv_dim = inputs[1].shape_[2];
  const index_t attn_batches      = inputs[2].shape_[0];
  const index_t q_seq_len         = inputs[2].shape_[1];
  const index_t embed_dim         = output_lin_kv_dim / 2;
  const index_t head_dim          = embed_dim / params.heads;
  const index_t lead_dim_kv       = attn_batches * head_dim * 2;
  const index_t batch_stride_kv   = 2 * head_dim;
  const float alpha               = 1.f;

  if (req[0] != kNullOp) {
    if (req[0] == kWriteTo) {
      memset(keys_values_grads, 0, outputs[0].shape_.Size() * sizeof (float));
    }
    const float beta = req[0] == kAddTo ? 1.f : 0.f;
    strided_batch_sgemm(false,
                        true,
                        head_dim,
                        kv_seq_len,
                        q_seq_len,
                        alpha,
                        output_grads,
                        head_dim * attn_batches,
                        head_dim,
                        attention_maps,
                        kv_seq_len,
                        kv_seq_len * q_seq_len,
                        beta,
                        keys_values_grads + head_dim,
                        lead_dim_kv,
                        batch_stride_kv,
                        attn_batches);
  }
  if (req[1] != kNullOp) {
    const float beta = req[1] == kAddTo ? 1.f : 0.f;
    strided_batch_sgemm(true,
                        false,
                        kv_seq_len,
                        q_seq_len,
                        head_dim,
                        alpha,
                        keys_values + head_dim,
                        lead_dim_kv,
                        batch_stride_kv,
                        output_grads,
                        head_dim * attn_batches,
                        head_dim,
                        beta,
                        attention_maps_grads,
                        kv_seq_len,
                        kv_seq_len * q_seq_len,
                        attn_batches);
  }
}

NNVM_REGISTER_OP(_contrib_interleaved_matmul_selfatt_qk)
.describe(R"code(Compute the matrix multiplication between the projections of
queries and keys in multihead attention use as self attention.

the input must be a single tensor of interleaved projections
of queries, keys and values following the layout:
(seq_length, batch_size, num_heads * head_dim * 3)

the equivalent code would be:
tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
q_proj = mx.nd.transpose(tmp[:,:,:,0,:], axes=(1, 2, 0, 3))
q_proj = mx.nd.reshape(q_proj, shape=(-1, 0, 0), reverse=True)
q_proj = mx.nd.contrib.div_sqrt_dim(q_proj)
k_proj = mx.nd.transpose(tmp[:,:,:,1,:], axes=(1, 2, 0, 3))
k_proj = mx.nd.reshap(k_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(q_proj, k_proj, transpose_b=True)
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", InterleavedMatMulSelfAttQKCPU)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseIn{"_backward_interleaved_matmul_selfatt_qk"})
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_qk)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<FCompute>("FCompute<cpu>", BackwardInterleavedMatMulSelfAttQKCPU);

NNVM_REGISTER_OP(_contrib_interleaved_matmul_selfatt_valatt)
.describe(R"code(Compute the matrix multiplication between the projections of
values and the attention weights in multihead attention use as self attention.

the inputs must be a tensor of interleaved projections
of queries, keys and values following the layout:
(seq_length, batch_size, num_heads * head_dim * 3)

and the attention weights following the layout:
(batch_size, seq_length, seq_length)

the equivalent code would be:
tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
v_proj = mx.nd.transpose(tmp[:,:,:,2,:], axes=(1, 2, 0, 3))
v_proj = mx.nd.reshape(v_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(attention, v_proj, transpose_b=True)
output = mx.nd.reshape(output, shape=(-1, num_heads, 0, 0), reverse=True)
output = mx.nd.transpose(output, axes=(0, 2, 1, 3))
output = mx.nd.reshape(output, shape=(0, 0, -1))
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulSelfAttValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", InterleavedMatMulSelfAttValAttCPU)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseIn{"_backward_interleaved_matmul_selfatt_valatt"})
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_valatt)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<FCompute>("FCompute<cpu>", BackwardInterleavedMatMulSelfAttValAttCPU);

NNVM_REGISTER_OP(_contrib_interleaved_matmul_encdec_qk)
.describe(R"code(Compute the matrix multiplication between the projections of
queries and keys in multihead attention use as encoder-decoder.

the inputs must be a tensor of projections of queries following the layout:
(seq_length, batch_size, num_heads * head_dim)

and a tensor of interleaved projections of values and keys following the layout:
(seq_length, batch_size, num_heads * head_dim * 2)

the equivalent code would be:
q_proj = mx.nd.transpose(queries, axes=(1, 2, 0, 3))
q_proj = mx.nd.reshape(q_proj, shape=(-1, 0, 0), reverse=True)
q_proj = mx.nd.contrib.div_sqrt_dim(q_proj)
tmp = mx.nd.reshape(keys_values, shape=(0, 0, num_heads, 2, -1))
k_proj = mx.nd.transpose(tmp[:,:,:,0,:], axes=(1, 2, 0, 3))
k_proj = mx.nd.reshap(k_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(q_proj, k_proj, transpose_b=True)
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries", "keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulEncDecQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", InterleavedMatMulEncDecQKCPU)
.set_attr<nnvm::FGradient>("FGradient",
    ElemwiseGradUseIn{"_backward_interleaved_matmul_encdec_qk"})
.add_argument("queries", "NDArray-or-Symbol", "Queries")
.add_argument("keys_values", "NDArray-or-Symbol", "Keys and values interleaved")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_encdec_qk)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<FCompute>("FCompute<cpu>", BackwardInterleavedMatMulEncDecQKCPU);

NNVM_REGISTER_OP(_contrib_interleaved_matmul_encdec_valatt)
.describe(R"code(Compute the matrix multiplication between the projections of
values and the attention weights in multihead attention use as encoder-decoder.

the inputs must be a tensor of interleaved projections of
keys and values following the layout:
(seq_length, batch_size, num_heads * head_dim * 2)

and the attention weights following the layout:
(batch_size, seq_length, seq_length)

the equivalent code would be:

tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
v_proj = mx.nd.transpose(tmp[:,:,:,1,:], axes=(1, 2, 0, 3))
v_proj = mx.nd.reshape(v_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(attention, v_proj, transpose_b=True)
output = mx.nd.reshape(output, shape=(-1, num_heads, 0, 0), reverse=True)
output = mx.nd.transpose(output, axes=(0, 2, 1, 3))
output = mx.nd.reshape(output, shape=(0, 0, -1))
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulEncDecValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", InterleavedMatMulEncDecValAttCPU)
.set_attr<nnvm::FGradient>("FGradient",
    ElemwiseGradUseIn{"_backward_interleaved_matmul_encdec_valatt"})
.add_argument("keys_values", "NDArray-or-Symbol", "Keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_encdec_valatt)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<FCompute>("FCompute<cpu>", BackwardInterleavedMatMulEncDecValAttCPU);


// relu
MXNET_OPERATOR_REGISTER_UNARY(_contrib_div_sqrt_dim)
.describe(R"code(Rescale the input by the square root of the channel dimension.

   out = data / sqrt(data.shape[-1])

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", DivSqrtDimForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_div_sqrt_dim"});

}  // namespace op
}  // namespace mxnet
