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

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
#if MXNET_USE_MKLDNN == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/storage.h>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include "../../math.h"
#include "../../math_functions-inl.h"
#include "../../operator_common.h"
#include "../../rnn_impl.h"
#include "../../rnn-inl.h"
#include "mkldnn.hpp"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

algorithm GetMKLDNNRNNAlgo(int mode,
                           int* ngates,
                           int* nstates) {
  algorithm algo = algorithm::vanilla_rnn;
  switch (mode) {
    case rnn_enum::kLstm:
      *ngates = 4;
      *nstates = 2;
      algo = algorithm::vanilla_lstm;
      break;
    case rnn_enum::kGru:
      *ngates = 3;
      *nstates = 1;
      algo = algorithm::vanilla_gru;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      *ngates = 1;
      *nstates = 1;
      algo = algorithm::vanilla_rnn;
      break;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      break;
  }
  return algo;
}

void ConcatData(mkldnn::memory::format src_format,
                mkldnn::memory::format dst_format,
                std::vector<mkldnn::memory::dims> srcs_cds,
                mkldnn::memory::dims dst_cds,
                mkldnn::memory::data_type mkldnn_dtype,
                int concat_dimension,
                std::vector<void*> srcs_data,
                const mkldnn::memory &dst) {
  auto cpu_engine = CpuEngine::Get()->get_engine();
  std::vector<mkldnn::memory::primitive_desc> srcs_pd;
  std::vector<mkldnn::memory> srcs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    auto desc = mkldnn::memory::desc(srcs_cds[i], mkldnn_dtype, src_format);
    auto mpd = mkldnn::memory::primitive_desc(desc, cpu_engine);
    auto src_memory = mkldnn::memory(mpd, srcs_data[i]);
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    inputs.push_back(srcs[i]);
  }
  auto dst_desc = mkldnn::memory::desc(dst_cds, mkldnn_dtype, dst_format);
  auto concat_pd = concat::primitive_desc(dst_desc, concat_dimension, srcs_pd);
  MKLDNNStream::Get()->RegisterPrim(concat(concat_pd, inputs, dst));
  MKLDNNStream::Get()->Submit();
}

inline size_t GetMKLDNNRNNCacheMemorySize(int L,
                                          int D,
                                          int T,
                                          int N,
                                          int I,
                                          int H,
                                          int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = 2 * (D * (I + H) * 4 * H + (L - 1) * D * (D * H + H) * 4 * H +
             L * D * 2 * N * H) + T * N * D * H + L * 2 * D * 4 * H + (L + 2) * D * 2 * N * H +
             6 * D * (I + H + 2) * 4 * H + T * N * I * 2;
      break;
    case rnn_enum::kGru:
      size = 2 * (D * (I + H) * 3 * H + (L - 1) * D * (D * H + H) * 3 * H +
             L * D * 2 * N * H) + T * N * D * H + L * 2 * D * 3 * H + (L + 2) * D * 2 * N * H +
             6 * D * (I + H + 2) * 3 * H + T * N * I * 2;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      size = 2 * (D * (I + H) * 1 * H + (L - 1) * D * (D * H + H) * 1 * H +
             L * D * 2 * N * H) + T * N * D * H + L * 2 * D * 1 * H + (L + 2) * D * 2 * N * H +
             6 * D * (I + H + 2) * 1 * H + T * N * I * 2;
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

template <typename DType>
void AdjustGruGateOrder(DType* weight,
                        const int I,
                        const int H) {
  // mxnet gru gate order is reset, update and new gates
  // mkldnn gru gate order is update, reset and new gates
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  DType* weight_reset = weight;
  DType* weight_update = weight + I * H;
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < I * H; i++) {
    DType tmp = weight_update[i];
    weight_update[i] = weight_reset[i];
    weight_reset[i] = tmp;
  }
}
// since there is different sematics of MKLDNN's Fused RNN and Mxnet FusedRNN,
// bidirectional will be fused layer by layer,
// unidirectional will be done by fused 1 + fused (L - 1) layers or fused L layers(when I = H)

template <typename DType>
void MKLDNNRNNForwardSingleLayerBi(bool state_outputs,
                                   const int T,
                                   const int N,
                                   const int I,
                                   const int H,
                                   DType* x_ptr,
                                   mkldnn::memory user_src_layer_memory,
                                   DType* hx_ptr,
                                   DType* cx_ptr,
                                   DType* w_ptr,
                                   DType* b_ptr,
                                   DType* y_ptr,
                                   DType* hy_ptr,
                                   DType* cy_ptr,
                                   std::vector<mkldnn::memory> *concat_weight_memory,
                                   std::vector<mkldnn::memory> *concat_iter_memory,
                                   std::vector<mkldnn::memory> *x_memory,
                                   std::vector<mkldnn::memory> *hcx_memory,
                                   std::vector<mkldnn::memory> *wx_memory,
                                   std::vector<mkldnn::memory> *wh_memory,
                                   std::vector<mkldnn::memory> *bias_memory,
                                   std::vector<mkldnn::memory> *y_memory,
                                   std::vector<mkldnn::memory> *hcy_memory,
                                   std::vector<primitive> *rnn_forward_prim,
                                   int layer_index,
                                   bool *has_cache,
                                   int lvalue,
                                   int dtype,
                                   bool is_train,
                                   int mode) {
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  DType* wx = w_ptr;  //  ngates * H, I
  DType* wh = w_ptr + I * H * ngates;  //  ngates * H, H
  DType* back_wx = w_ptr + ngates * H * (I + H);
  DType* back_wh = back_wx + I * H * ngates;
  DType* bx = b_ptr;
  DType* bh = b_ptr + H * ngates;
  DType* back_bx = b_ptr + single_b_size * 2;
  DType* back_bh = back_bx + H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  bool cached = *has_cache;
  mkldnn::memory::dims src_layer_tz = {T, N, I};
  mkldnn::memory::dims dst_layer_tz = {T, N, 2 * H};
  mkldnn::memory::dims weights_layer_tz = {1, 2, I, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims weights_iter_tz = {1, 2, H, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims bias_tz = {1, 2, ngates, H};
  mkldnn::memory::dims src_iter_tz = {1, 2, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims dst_iter_tz = {1, 2, nstates, N, H};  //  ldsnc

  std::vector<float> weights_scales(ngates * H);
  if (!cached) {
    if (mode == rnn_enum::kGru) {
      AdjustGruGateOrder(wx, I, H);
      AdjustGruGateOrder(back_wx, I, H);
      AdjustGruGateOrder(wh, H, H);
      AdjustGruGateOrder(back_wh, H, H);
    }
    auto src_wx = (*concat_weight_memory)[2 * layer_index];
    auto src_wh = (*concat_weight_memory)[2 * layer_index + 1];
    std::vector<void*> srcs_data1;
    srcs_data1.push_back(wx);
    srcs_data1.push_back(back_wx);
    ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
        {weights_layer_r_tz, weights_layer_r_tz}, weights_layer_tz,
        mkldnn_dtype, 1, srcs_data1, src_wx);
    srcs_data1.clear();
    srcs_data1.push_back(wh);
    srcs_data1.push_back(back_wh);
    ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
        {weights_iter_r_tz, weights_iter_r_tz}, weights_iter_tz,
         mkldnn_dtype, 1, srcs_data1, src_wh);
    int tmpvalue = 0;
    if (lvalue > 0) {
      tmpvalue = lvalue + 1;
    }
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wx, (*wx_memory)[tmpvalue]));
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wh, (*wh_memory)[tmpvalue]));

    DType* user_bias = reinterpret_cast<DType *>
        ((*bias_memory)[tmpvalue].get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < single_b_size; j++) {
      user_bias[j] = bx[j] + bh[j];
      user_bias[single_b_size + j] = back_bx[j] + back_bh[j];
    }
  }
  if (lvalue > 0) {
    (*wx_memory)[layer_index].set_data_handle((*wx_memory)[lvalue + 1].get_data_handle());
    (*wh_memory)[layer_index].set_data_handle((*wh_memory)[lvalue + 1].get_data_handle());
    (*bias_memory)[layer_index].set_data_handle((*bias_memory)[lvalue + 1].get_data_handle());
  }

  auto src_layer_md = mkldnn::memory::desc(
      { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto weight_layer_md = mkldnn::memory::desc(
      { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto weight_iter_md = mkldnn::memory::desc(
      { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto dst_layer_md = mkldnn::memory::desc(
      { dst_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_iter_md = mkldnn::memory::desc(
      { dst_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto src_iter_md = mkldnn::memory::desc(
      {src_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto bias_md = mkldnn::memory::desc({bias_tz},
      mkldnn_dtype, mkldnn::memory::format::ldgo);

  auto user_src_iter_memory = (*concat_iter_memory)[2];
  if (mode == rnn_enum::kLstm) {
    std::vector<void*> srcs_data1;
    srcs_data1.push_back(hx_ptr);
    srcs_data1.push_back(cx_ptr);
    auto tmp1_src_iter_memory = (*concat_iter_memory)[0];
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data1, tmp1_src_iter_memory);
    std::vector<void*> srcs_data2;
    srcs_data2.push_back(hx_ptr + single_cell_size);
    srcs_data2.push_back(cx_ptr + single_cell_size);
    auto tmp2_src_iter_memory = (*concat_iter_memory)[1];
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data2, tmp2_src_iter_memory);
    std::vector<void*> srcs_data3;
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp1_src_iter_memory.get_data_handle()));
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp2_src_iter_memory.get_data_handle()));
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, nstates, N, H}, {1, 1, nstates, N, H}}, {1, 2, nstates, N, H},
        mkldnn_dtype, 1, srcs_data3, user_src_iter_memory);
  } else {
    user_src_iter_memory.set_data_handle(hx_ptr);
  }
  (*hcx_memory)[layer_index].set_data_handle(user_src_iter_memory.get_data_handle());

  rnn_cell::desc rnn_cell(nalgorithm,
      mode == rnn_enum::kRnnRelu ? algorithm::eltwise_relu : algorithm::eltwise_tanh);

  rnn_forward::desc layer_desc(prop_kind::forward_inference, rnn_cell,
      rnn_direction::bidirectional_concat, src_layer_md,
      src_iter_md, weight_layer_md, weight_iter_md,
      bias_md, dst_layer_md, dst_iter_md);

  auto prim_desc
       = rnn_forward::primitive_desc(layer_desc, cpu_engine);

  if (x_ptr && layer_index == 0) {
    (*x_memory)[layer_index].set_data_handle(x_ptr);
  } else {
    (*x_memory)[layer_index].set_data_handle(user_src_layer_memory.get_data_handle());
  }
  (*y_memory)[layer_index].set_data_handle(y_ptr);

  if (rnn_forward_prim->size() <= layer_index) {
    primitive rnn_prim = rnn_forward(prim_desc, (*x_memory)[layer_index],
          (*hcx_memory)[layer_index], (*wx_memory)[layer_index],
          (*wh_memory)[layer_index], (*bias_memory)[layer_index],
          (*y_memory)[layer_index],
         (*hcy_memory)[layer_index], null_memory_);
    rnn_forward_prim->push_back(rnn_prim);
  }
  MKLDNNStream::Get()->RegisterPrim((*rnn_forward_prim)[layer_index]);
  MKLDNNStream::Get()->Submit();

  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *> ((*hcy_memory)[layer_index].get_data_handle());
    if (mode == rnn_enum::kLstm) {
      offset1 = nstates * single_cell_size;
      offset2 = (nstates + 1) * single_cell_size;
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
        hy_ptr[n + single_cell_size] = dst_hcy[n + offset1];
        cy_ptr[n] = dst_hcy[n + single_cell_size];
        cy_ptr[n + single_cell_size] = dst_hcy[n + offset2];
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < 2 * single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
      }
    }
  }
}


template <typename DType>
void MKLDNNRNNForwardUnidi(bool state_outputs,
                           const int L,
                           const int T,
                           const int N,
                           const int I,
                           const int H,
                           DType* x_ptr,
                           mkldnn::memory user_src_layer_memory,
                           DType* hx_ptr,
                           DType* cx_ptr,
                           DType* w_ptr,
                           DType* b_ptr,
                           DType* y_ptr,
                           DType* hy_ptr,
                           DType* cy_ptr,
                           std::vector<mkldnn::memory> *concat_weight_memory,
                           std::vector<mkldnn::memory> *concat_iter_memory,
                           std::vector<mkldnn::memory> *x_memory,
                           std::vector<mkldnn::memory> *hcx_memory,
                           std::vector<mkldnn::memory> *wx_memory,
                           std::vector<mkldnn::memory> *wh_memory,
                           std::vector<mkldnn::memory> *bias_memory,
                           std::vector<mkldnn::memory> *y_memory,
                           std::vector<mkldnn::memory> *hcy_memory,
                           std::vector<primitive> *rnn_forward_prim,
                           int layer_index,
                           bool *has_cache,
                           int dtype,
                           bool is_train,
                           int mode) {
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int cell_size = N * H;
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  int w_size = (I + H) * H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  bool cached = *has_cache;

  mkldnn::memory::dims src_layer_tz = {T, N, I};
  mkldnn::memory::dims dst_layer_tz = {T, N, H};
  mkldnn::memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
  mkldnn::memory::dims bias_tz = {L, 1, ngates, H};
  mkldnn::memory::dims src_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims dst_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder

  auto weight_layer_md = mkldnn::memory::desc(
      { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto weight_iter_md = mkldnn::memory::desc(
      { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto src_layer_md = mkldnn::memory::desc(
      { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_layer_md = mkldnn::memory::desc(
      {dst_layer_tz}, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto src_iter_md = mkldnn::memory::desc(
      {src_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto bias_md = mkldnn::memory::desc({bias_tz},
      mkldnn_dtype, mkldnn::memory::format::ldgo);
  auto dst_iter_md = mkldnn::memory::desc(
      {dst_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);

  for (int l = 0; l < L; l++) {
    if (mode == rnn_enum::kLstm) {
      std::vector<void*> srcs_data;
      srcs_data.push_back(hx_ptr);
      srcs_data.push_back(cx_ptr);
      auto tmp_src_iter_memory = (*concat_iter_memory)[l + layer_index];
      ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
          {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype,
          2, srcs_data, tmp_src_iter_memory);
    } else {
      (*concat_iter_memory)[l + layer_index].set_data_handle(hx_ptr);
    }
    hx_ptr += cell_size;
    if (mode == rnn_enum::kLstm) {
      cx_ptr += cell_size;
    }
  }

  auto user_src_iter_memory = null_memory_;
  if (L == 1) {
    user_src_iter_memory = (*concat_iter_memory)[layer_index];
  } else {
    user_src_iter_memory = (*concat_iter_memory)[L + layer_index];
    std::vector<void*> src_l_data;
    std::vector<mkldnn::memory::dims> src_l_dim;
    for (int l = 0; l < L; l++) {
      src_l_data.push_back(reinterpret_cast<DType *>
          ((*concat_iter_memory)[l + layer_index].get_data_handle()));
      src_l_dim.push_back({1, 1, nstates, N, H});
    }
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc, src_l_dim,
        {L, 1, nstates, N, H}, mkldnn_dtype, 0, src_l_data, user_src_iter_memory);
  }
  (*hcx_memory)[layer_index].set_data_handle(user_src_iter_memory.get_data_handle());

  auto src_wx_f = (*concat_weight_memory)[2 * layer_index];
  auto src_wh_f = (*concat_weight_memory)[2 * layer_index + 1];

  std::vector<void*> srcs_data_x;
  std::vector<void*> srcs_data_h;
  std::vector<mkldnn::memory::dims> src_l_dim_x;
  std::vector<mkldnn::memory::dims> src_l_dim_h;
  std::vector<float> weights_scales(ngates * H);
  if (!cached) {
    if (L == 1) {
      DType* wx = w_ptr;
      DType* wh = w_ptr + I * H * ngates;
      if (mode == rnn_enum::kGru) {
        AdjustGruGateOrder(wx, I, H);
        AdjustGruGateOrder(wh, H, H);
      }
      src_wx_f.set_data_handle(wx);
      src_wh_f.set_data_handle(wh);
    } else {
      for (int l = 0; l < L; l++) {
        DType* wx = w_ptr;
        DType* wh = w_ptr + I * H * ngates;
        if (mode == rnn_enum::kGru) {
          AdjustGruGateOrder(wx, I, H);
          AdjustGruGateOrder(wh, H, H);
        }
        srcs_data_x.push_back(wx);
        srcs_data_h.push_back(wh);
        src_l_dim_x.push_back(weights_layer_r_tz);
        src_l_dim_h.push_back(weights_iter_r_tz);
        w_ptr = w_ptr + w_size;
      }
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x, src_wx_f);
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h, src_wh_f);
    }
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wx_f, (*wx_memory)[layer_index]));
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wh_f, (*wh_memory)[layer_index]));

    DType* user_bias_f = reinterpret_cast<DType *> ((*bias_memory)[layer_index].get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < L * single_b_size; j++) {
      int k = j / single_b_size;
      user_bias_f[j] = b_ptr[j + k * single_b_size] + b_ptr[j + k * single_b_size + single_b_size];
    }
  }

  rnn_cell::desc rnn_cell(nalgorithm,
      mode == rnn_enum::kRnnRelu ? algorithm::eltwise_relu : algorithm::eltwise_tanh);

  rnn_forward::desc layer_desc(prop_kind::forward_inference, rnn_cell,
      rnn_direction::unidirectional, src_layer_md,
      src_iter_md, weight_layer_md, weight_iter_md,
      bias_md, dst_layer_md, dst_iter_md);

  auto prim_desc
       = rnn_forward::primitive_desc(layer_desc, cpu_engine);

  if (x_ptr && layer_index == 0) {
    (*x_memory)[layer_index].set_data_handle(x_ptr);
  } else {
    (*x_memory)[layer_index].set_data_handle(user_src_layer_memory.get_data_handle());
  }
  (*y_memory)[layer_index].set_data_handle(y_ptr);

  if (rnn_forward_prim->size() <= layer_index) {
    primitive rnn_prim = rnn_forward(prim_desc, (*x_memory)[layer_index],
          (*hcx_memory)[layer_index], (*wx_memory)[layer_index],
          (*wh_memory)[layer_index], (*bias_memory)[layer_index],
          (*y_memory)[layer_index],
         (*hcy_memory)[layer_index], null_memory_);
    rnn_forward_prim->push_back(rnn_prim);
  }
  MKLDNNStream::Get()->RegisterPrim((*rnn_forward_prim)[layer_index]);
  MKLDNNStream::Get()->Submit();

  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *> ((*hcy_memory)[layer_index].get_data_handle());
    if (mode == rnn_enum::kLstm) {
      for (int l = 0; l < L; l++) {
        offset1 = l * single_cell_size;
        offset2 = l * nstates * single_cell_size;
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[offset1 + n] = dst_hcy[offset2 + n];
          cy_ptr[offset1 + n] = dst_hcy[offset2 + n + single_cell_size];
        }
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < L * single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
      }
    }
  }
}

template <typename DType>
void MKLDNNRNNForward(bool state_outputs,
                      const int L,
                      const int D,
                      const int T,
                      const int N,
                      const int I,
                      const int H,
                      DType* x_ptr,
                      DType* hx_ptr,
                      DType* cx_ptr,
                      DType* w_ptr,
                      DType* b_ptr,
                      DType* y_ptr,
                      DType* hy_ptr,
                      DType* cy_ptr,
                      std::vector<mkldnn::memory> *concat_weight_memory,
                      std::vector<mkldnn::memory> *concat_iter_memory,
                      std::vector<mkldnn::memory> *x_memory,
                      std::vector<mkldnn::memory> *hcx_memory,
                      std::vector<mkldnn::memory> *wx_memory,
                      std::vector<mkldnn::memory> *wh_memory,
                      std::vector<mkldnn::memory> *bias_memory,
                      std::vector<mkldnn::memory> *y_memory,
                      std::vector<mkldnn::memory> *hcy_memory,
                      std::vector<primitive> *rnn_forward_prim,
                      bool *has_cache,
                      int dtype,
                      bool is_train,
                      int mode) {
  int ngates = 0, nstates = 0;
  GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  DType* tmpNull = NULL;
  // when D = 1 and I == H, L layers can be fused together
  if (D == 1 && I == H) {
    MKLDNNRNNForwardUnidi(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
        concat_iter_memory, x_memory, hcx_memory, wx_memory, wh_memory,
        bias_memory, y_memory, hcy_memory, rnn_forward_prim,
        0, has_cache, dtype, is_train, mode);
  } else {
    auto user_src_layer_memory_l = null_memory_;
    if (D == 2) {
      MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, I, H, x_ptr, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
          concat_iter_memory, x_memory, hcx_memory, wx_memory, wh_memory,
          bias_memory, y_memory, hcy_memory, rnn_forward_prim,
          0, has_cache, 0, dtype, is_train, mode);
    } else {
      MKLDNNRNNForwardUnidi(state_outputs, 1, T, N, I, H, x_ptr, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
          concat_iter_memory, x_memory, hcx_memory, wx_memory, wh_memory,
          bias_memory, y_memory, hcy_memory, rnn_forward_prim,
          0, has_cache, dtype, is_train, mode);
    }
    if (L > 1) {
      user_src_layer_memory_l = (*y_memory)[0];
      //  go to next L - 1 layers.
      //  If D = 2, do it layer by layer. If D = 1, fused L - 1 layers
      w_ptr += w_size;
      b_ptr += b_size;
      if (D == 2) {
        w_size = (H * D + H) * H * ngates * D;
        for (int l = 0; l < L - 1; l++) {
          if (state_outputs) {
            hy_ptr += cell_size;
            if (mode == rnn_enum::kLstm) {
              cy_ptr += cell_size;
            }
          }
          hx_ptr += cell_size;
          if (mode == rnn_enum::kLstm) {
            cx_ptr += cell_size;
          }
          MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, D * H, H, tmpNull,
              user_src_layer_memory_l, hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr,
              cy_ptr, concat_weight_memory, concat_iter_memory, x_memory,
              hcx_memory, wx_memory, wh_memory, bias_memory,
              y_memory, hcy_memory, rnn_forward_prim,
              1, has_cache, l + 1, dtype, is_train, mode);
          user_src_layer_memory_l = (*y_memory)[1];
          w_ptr += w_size;
          b_ptr += b_size;
        }
      }
      if (D == 1) {
        w_size = (H + H) * H * ngates;
        MKLDNNRNNForwardUnidi(state_outputs, L - 1, T, N, H, H, tmpNull, user_src_layer_memory_l,
            hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
            concat_iter_memory, x_memory, hcx_memory, wx_memory,
            wh_memory, bias_memory, y_memory, hcy_memory,
            rnn_forward_prim, 1, has_cache, dtype, is_train, mode);
      }
    }
  }
  *has_cache = true;
}

template <typename DType>
void MKLDNNRNNForwardInference(bool state_outputs,
                               const int num_layers,
                               const int direction,
                               const int seq_length,
                               const int batch_size,
                               const int input_size,
                               const int state_size,
                               DType* x_ptr,
                               DType* hx_ptr,
                               DType* cx_ptr,
                               DType* w_ptr,
                               DType* b_ptr,
                               DType* y_ptr,
                               DType* hy_ptr,
                               DType* cy_ptr,
                               std::vector<mkldnn::memory>* concat_weight_memory,
                               std::vector<mkldnn::memory>* concat_iter_memory,
                               std::vector<mkldnn::memory> *x_memory,
                               std::vector<mkldnn::memory> *hcx_memory,
                               std::vector<mkldnn::memory> *wx_memory,
                               std::vector<mkldnn::memory> *wh_memory,
                               std::vector<mkldnn::memory> *bias_memory,
                               std::vector<mkldnn::memory> *y_memory,
                               std::vector<mkldnn::memory> *hcy_memory,
                               std::vector<primitive> *rnn_forward_prim,
                               bool *has_cache,
                               int dtype,
                               bool is_train,
                               int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
    case rnn_enum::kGru:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      MKLDNNRNNForward<DType>(state_outputs, num_layers, direction, seq_length,
                              batch_size, input_size, state_size, x_ptr, hx_ptr,
                              cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
                              concat_weight_memory, concat_iter_memory, x_memory,
                              hcx_memory, wx_memory, wh_memory,
                              bias_memory, y_memory, hcy_memory, rnn_forward_prim,
                              has_cache, dtype, is_train, mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
