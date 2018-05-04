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
 * Copyright (c) 2015 by Contributors
 * \file    rnn_impl.hpp
 * \brief
 * \author  Shu Zhang(shu.zhang@intel.com)
*/
#ifndef MXNET_OPERATOR_RNN_IMPL_HPP_
#define MXNET_OPERATOR_RNN_IMPL_HPP_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./linalg.h"

#define UNIDIRECT 1
#define BIDIRECT 2

template<typename DType>
inline DType sigmoid(DType x) {
  return 1.0f / (1.0f + exp(-x));
}

template<typename DType>
void GruForwardInferenceSingleLayer(DType* ws,
                                    DType* tmp_buf,
                                    bool state_outputs,
                                    const int D,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    const Tensor<cpu, 2, DType> &x,
                                    const Tensor<cpu, 2, DType> &hx,
                                    DType* wx_ptr,
                                    DType* wh_ptr,
                                    DType* bx_ptr,
                                    DType* bh_ptr,
                                    DType* y_ptr,
                                    DType* hy_ptr) {
  DType* ht = y_ptr;
  DType* ht_1 = y_ptr;
  DType* back_ht_1 = y_ptr + (T-1) * N * H * D + H;
  DType* back_ht = back_ht_1;
  DType* gemmC1  = ws;              // [D, T, N, 3 * H]
  DType* gemmC2  = gemmC1 + D * T * N * 3 * H;  // N * 3 * H
  DType* rt = gemmC2 + N * 3 * H;
  DType* zt = rt + N * H;
  DType* nt = zt + N * H;
  DType* back_wx_ptr = wx_ptr + I * 3 * H + H * 3 * H;
  DType* back_wh_ptr = wh_ptr + I * 3 * H + H * 3 * H;
  DType* back_bx_ptr = (bx_ptr != NULL)? bx_ptr + 3 * H * 2 : NULL;
  DType* back_bh_ptr = (bh_ptr != NULL)? bh_ptr + 3 * H * 2: NULL;
  DType* back_gemmC1 = gemmC1 + T * N * 3 * H;
  DType* gemmC1_t = gemmC1;

  const Tensor<cpu, 2, DType> wx(wx_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh(wh_ptr, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> bx(bx_ptr, Shape2(3, H));
  const Tensor<cpu, 2, DType> bh(bh_ptr, Shape2(3, H));
  const Tensor<cpu, 2, DType> back_wx(back_wx_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> back_wh(back_wh_ptr, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> back_bx(back_bx_ptr, Shape2(3, H));
  const Tensor<cpu, 2, DType> back_bh(back_bh_ptr, Shape2(3, H));

  if (D == UNIDIRECT) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int j = 0; j < H; j++) {
        y_ptr[i * H + j] = hx[i][j];
      }
  } else {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int j = 0; j < H; j++) {
        y_ptr[i * D * H + j] = hx[i][j];
        back_ht_1[i * D * H + j] = hx[N + i][j];
    }
  }
  Tensor<cpu, 2, DType> dgemmC1(ws, Shape2(T * N, 3 * H));
  Tensor<cpu, 2, DType> dgemmC2(gemmC2, Shape2(N, 3 * H));
  Tensor<cpu, 2, DType> dback_gemmC1(back_gemmC1, Shape2(T * N, 3 * H));

  // x * wx.T : [T * N, I] * [I, 3 * H]
  DType alpha = 1.0;
  DType beta = 0.0;
  linalg_gemm(x, wx, dgemmC1, alpha, beta, false, true);
  if (D == BIDIRECT) {
    linalg_gemm(x, back_wx, dback_gemmC1, alpha, beta, false, true);
  }

  for (int t = 0; t < T; t++) {
    //  perform the first direction, X * wx and H * wh for each step
    //  ht-1 * wh, ht-1:[N, H] wh:[3 * H, H]
    Tensor<cpu, 2, DType> dht_1(ht_1, Shape2(N, D * H));
    if (D == UNIDIRECT) {
      linalg_gemm(dht_1, wh, dgemmC2, alpha, beta, false, true);
    } else {
      Tensor<cpu, 3, DType> dht_1_tmp = Tensor<cpu, 3, DType>(reinterpret_cast<DType*>(tmp_buf),
                                     Shape3(D, H, N));
      dht_1_tmp = reshape(dht_1.T(), Shape3(D, H, N));
      linalg_gemm(dht_1_tmp[0], wh, dgemmC2, alpha, beta, true, true);
    }
    gemmC1_t = gemmC1 + t * N * 3 * H;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        int rtb = i * 3 * H;
        int ztb = i * 3 * H + H;
        int ntb = i * 3 * H + 2 * H;
        rt[i * H + j] = sigmoid(gemmC1_t[rtb + j] + gemmC2[rtb + j]
            + bx[0][j] + bh[0][j]);
        zt[i * H + j] = sigmoid(gemmC1_t[ztb + j] + gemmC2[ztb + j]
            + bx[1][j] + bh[1][j]);
        nt[i * H + j] = tanh(gemmC1_t[ntb + j] + bx[2][j] +
            rt[i * H + j] * (gemmC2[ntb + j] + bh[2][j]));
        ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +
            zt[i * H + j] * ht_1[i * D * H + j];
      }
    }
    ht_1 = ht;
    ht = ht + D * H * N;
    //  perform the second direction
    if (D == BIDIRECT) {
      gemmC1_t = back_gemmC1 + (T - 1 - t) * N * 3 * H;
      Tensor<cpu, 2, DType> dback_ht_1(back_ht_1, Shape2(N, D * H));
      Tensor<cpu, 3, DType> dback_ht_1_tmp = Tensor<cpu, 3, DType>
          (reinterpret_cast<DType*>(tmp_buf), Shape3(D, H, N));
      dback_ht_1_tmp = reshape(dback_ht_1.T(), Shape3(D, H, N));
      linalg_gemm(dback_ht_1_tmp[0], back_wh, dgemmC2, alpha, beta, true, true);

      #pragma omp parallel for
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < H; ++j) {
          int rtb = i * 3 * H;
          int ztb = i * 3 * H + H;
          int ntb = i * 3 * H + 2 * H;
          rt[i * H + j] = sigmoid(gemmC1_t[rtb + j] +
              gemmC2[rtb + j] + back_bx[0][j] + back_bh[0][j]);
          zt[i * H + j] = sigmoid(gemmC1_t[ztb + j] +
              gemmC2[ztb + j] + back_bx[1][j]+ back_bh[1][j]);
          nt[i * H + j] = tanh(gemmC1_t[ntb + j] + back_bx[2][j]
              + rt[i * H + j] * (gemmC2[ntb + j] + back_bh[2][j]));
          back_ht[i * D * H + j] = (1 - zt[i * H + j]) * nt[i * H + j]
              + zt[i * H + j] * back_ht_1[i * D * H + j];
        }
      }
      back_ht_1 = back_ht;
      back_ht = back_ht - D * H * N;
    }
  }
  //  copy last state to hy, from(N, H * D) to (D, N, H)
  if (state_outputs) {
    if (D == UNIDIRECT) {
      DType* y_start = y_ptr + (T - 1) * N * H;
      #pragma omp parallel for
      for (int i = 0; i < N; i++)
        for (int j = 0; j < H; j++) {
          hy_ptr[i * H + j] = y_start[i * H + j];
        }
    } else {
      DType* y_start = y_ptr + (T - 1) * N * H * D;
      DType* y_back_start = y_ptr + H;
      #pragma omp parallel for
      for (int i = 0; i < N; i++)
        for (int j = 0; j < H; j++) {
          hy_ptr[i * H + j] = y_start[i * D * H + j];
          hy_ptr[N * H + i * H + j] = y_back_start[i * D * H + j];
        }
    }
  }
}

template <typename DType>
void GruForwardInference(DType* ws,
                         bool state_outputs,
                         const int L,
                         const int D,
                         const int T,
                         const int N,
                         int I,
                         const int H,
                         DType* x_ptr,
                         DType* hx_ptr,
                         DType* w_ptr,
                         DType* y_ptr,
                         DType* hy_ptr) {
  DType* wx = w_ptr;
  DType* wh = wx + I * H * 3;
  DType* bx = wh + H * H * 3 + (D - 1) * (H * H * 3 + I * H * 3)
      + (L - 1) * ((D + 1) * H) * H * 3 * D;
  DType* bh = bx + H * 3;

  DType* y_tmp = ws;
  DType* y_l = x_ptr;
  DType* tmp_buf = y_tmp + D * T * N * H;
  DType* ws2 = y_tmp + D * T * N * H + D * H * N;

  DType* wx_l = wx;
  DType* wh_l = wh;
  DType* bx_l = bx;
  DType* bh_l = bh;
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(D * L, N, H));
  DType* hy_l = hy_ptr;
  for (int l = 0; l < L; l++) {
    Tensor<cpu, 2, DType> x_l(y_l, Shape2(T * N, I));
    if ((L + l) % 2) {
      y_l = y_ptr;
    } else {
      y_l = y_tmp;
    }
    Tensor<cpu, 2, DType> hx_l = hx[D * l];
    GruForwardInferenceSingleLayer<DType>(ws2, tmp_buf, state_outputs, D, T, N, I, H,
                                        x_l, hx_l, wx_l, wh_l, bx_l, bh_l, y_l, hy_l);
    hy_l = hy_l + D * N * H;
    bx_l = bx_l + 3 * H * D * 2;
    bh_l = bh_l + 3 * H * D * 2;
    wx_l = wx_l + I * H * 3 * D + H * H * 3 * D;
    if (l == 0) {
      I = D * H;
    }
    wh_l = wx_l + I * 3 * H;
  }
}


template<typename DType>
void GruForwardTrainingSingleLayer(DType* ws,
                                   DType* tmp_buf,
                                   bool state_outputs,
                                   const int D,
                                   const int T,
                                   const int N,
                                   const int I,
                                   const int H,
                                   const Tensor<cpu, 2, DType> &x,
                                   const Tensor<cpu, 2, DType> &hx,
                                   DType* wx_ptr,
                                   DType* wh_ptr,
                                   DType* bx_ptr,
                                   DType* bh_ptr,
                                   DType* gateR,
                                   DType* gateZ,
                                   DType* gateN,
                                   DType* Mnh,
                                   DType* y_ptr,
                                   DType* hy_ptr) {
  DType* ht = y_ptr;
  DType* ht_1 = y_ptr;
  DType* back_ht_1 = y_ptr + (T - 1)* N * H * D + H;
  DType* back_ht = back_ht_1;

  DType* gemmC1  = ws;              // [D, T, N, 3 * H]
  DType* gemmC2  = gemmC1 + D * T * N * 3 * H;  // N * 3 * H
  DType* rt = gateR;
  DType* zt = gateZ;
  DType* nt = gateN;
  DType* back_wx_ptr = wx_ptr + I * 3 * H + H * 3 * H;
  DType* back_wh_ptr = wh_ptr + I * 3 * H + H * 3 * H;
  DType* back_bx_ptr = (bx_ptr != NULL)? bx_ptr + 3 * H * 2 : NULL;
  DType* back_bh_ptr = (bh_ptr != NULL)? bh_ptr + 3 * H * 2 : NULL;
  DType* back_gateR = gateR + T * N * H;
  DType* back_gateZ = gateZ + T * N * H;
  DType* back_gateN = gateN + T * N * H;
  DType* back_Mnh = Mnh + T * N * H;
  DType* back_gemmC1 = gemmC1 + T * N * 3 * H;
  DType* gemmC1_t = gemmC1;

  const Tensor<cpu, 2, DType> wx(wx_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh(wh_ptr, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> bx(bx_ptr, Shape2(3, H));
  const Tensor<cpu, 2, DType> bh(bh_ptr, Shape2(3, H));
  const Tensor<cpu, 2, DType> back_wx(back_wx_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> back_wh(back_wh_ptr, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> back_bx(back_bx_ptr, Shape2(3, H));
  const Tensor<cpu, 2, DType> back_bh(back_bh_ptr, Shape2(3, H));

  if (D == UNIDIRECT) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int j = 0; j < H; j++) {
        y_ptr[i * H + j] = hx[i][j];
      }
  } else {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int j = 0; j < H; j++) {
        y_ptr[i * D * H + j] = hx[i][j];
        back_ht_1[i * D * H + j] = hx[N + i][j];
    }
  }

  Tensor<cpu, 2, DType> dgemmC1(ws, Shape2(T * N, 3 * H));
  Tensor<cpu, 2, DType> dgemmC2(gemmC2, Shape2(N, 3 * H));
  Tensor<cpu, 2, DType> dback_gemmC1(back_gemmC1, Shape2(T * N, 3 * H));

  // x * wx.T : [T * N, I] * [I, 3 * H]
  DType alpha = 1.0;
  DType beta = 0.0;
  linalg_gemm(x, wx, dgemmC1, alpha, beta, false, true);
  if (D == BIDIRECT) {
    linalg_gemm(x, back_wx, dback_gemmC1, alpha, beta, false, true);
  }

  for (int t = 0; t < T; t++) {
    //  perform the first direction, X * wx and H * wh for each step
    //  ht-1 * wh, ht-1:[N, H] wh:[3 * H, H]
    Tensor<cpu, 2, DType> dht_1(ht_1, Shape2(N, D * H));
    if (D == UNIDIRECT) {
      linalg_gemm(dht_1, wh, dgemmC2, alpha, beta, false, true);
    } else {
      Tensor<cpu, 3, DType> dht_1_tmp = Tensor<cpu, 3, DType>(reinterpret_cast<DType*>(tmp_buf),
                                     Shape3(D, H, N));
      dht_1_tmp = reshape(dht_1.T(), Shape3(D, H, N));
      linalg_gemm(dht_1_tmp[0], wh, dgemmC2, alpha, beta, true, true);
    }
    gemmC1_t = gemmC1 + t * N * 3 * H;

    rt = gateR + t * N * H;
    zt = gateZ + t * N * H;
    nt = gateN + t * N * H;
    gemmC1_t = gemmC1 + t * N * 3 * H;
    DType* Mnht = Mnh + t * N * H;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        int rtb = i * 3 * H;
        int ztb = i * 3 * H + H;
        int ntb = i * 3 * H + 2 * H;
        Mnht[i * H + j] = gemmC2[ntb + j] + bh[2][j];
        rt[i * H + j] = sigmoid(gemmC1_t[rtb + j] + gemmC2[rtb + j]
            + bx[0][j] + bh[0][j]);
        zt[i * H + j] = sigmoid(gemmC1_t[ztb + j] + gemmC2[ztb + j]
            + bx[1][j] + bh[1][j]);
        nt[i * H + j] = tanh(gemmC1_t[ntb + j] + bx[2][j] +
            rt[i * H + j] * (gemmC2[ntb + j] + bh[2][j]));
        ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +
            zt[i * H + j] * ht_1[i * D * H + j];
      }
    }
    ht_1 = ht;
    ht = ht + D * H * N;
    //  perform the second direction
    if (D == BIDIRECT) {
      rt = back_gateR + (T - 1 - t) * N * H;
      zt = back_gateZ + (T - 1 - t) * N * H;
      nt = back_gateN + (T - 1 - t) * N * H;
      gemmC1_t = back_gemmC1 + (T - 1 - t) * N * 3 * H;
      Tensor<cpu, 2, DType> dback_ht_1(back_ht_1, Shape2(N, D * H));
      Tensor<cpu, 3, DType> dback_ht_1_tmp = Tensor<cpu, 3, DType>
          (reinterpret_cast<DType*>(tmp_buf), Shape3(D, H, N));
      dback_ht_1_tmp = reshape(dback_ht_1.T(), Shape3(D, H, N));
      linalg_gemm(dback_ht_1_tmp[0], back_wh, dgemmC2, alpha, beta, true, true);

      DType* back_Mnht = back_Mnh + (T - 1 - t) * N * H;
      #pragma omp parallel for
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < H; ++j) {
          int rtb = i * 3 * H;
          int ztb = i * 3 * H + H;
          int ntb = i * 3 * H + 2 * H;
          back_Mnht[i * H + j] = gemmC2[ntb + j] + back_bh[2][j];
          rt[i * H + j] = sigmoid(gemmC1_t[rtb + j] +
              gemmC2[rtb + j] + back_bx[0][j] + back_bh[0][j]);
          zt[i * H + j] = sigmoid(gemmC1_t[ztb + j] +
              gemmC2[ztb + j] + back_bx[1][j] + back_bh[1][j]);
          nt[i * H + j] = tanh(gemmC1_t[ntb + j] + back_bx[2][j]
              + rt[i * H + j] * (gemmC2[ntb + j] + back_bh[2][j]));
          back_ht[i * D * H + j] = (1 - zt[i * H + j]) * nt[i * H + j]
              + zt[i * H + j] * back_ht_1[i * D * H + j];
        }
      }
      back_ht_1 = back_ht;
      back_ht = back_ht - D * H * N;
    }
  }

  //  copy last state to hy, from(N, H * D) to (D, N, H)
  if (state_outputs) {
    if (D == UNIDIRECT) {
      DType* y_start = y_ptr + (T - 1) * N * H;
      #pragma omp parallel for
      for (int i = 0; i < N; i++)
        for (int j = 0; j < H; j++) {
          hy_ptr[i * H + j] = y_start[i * H + j];
        }
    } else {
      DType* y_start = y_ptr + (T - 1) * N * H * D;
      DType* y_back_start = y_ptr + H;
      #pragma omp parallel for
      for (int i = 0; i < N; i++)
        for (int j = 0; j < H; j++) {
          hy_ptr[i * H + j] = y_start[i * D * H + j];
          hy_ptr[N * H + i * H + j] = y_back_start[i * D * H + j];
        }
    }
  }
}

template <typename DType>
void GruForwardTraining(DType* ws,
                        DType* rs,
                        bool state_outputs,
                        const int L,
                        const int D,
                        const int T,
                        const int N,
                        int I,
                        const int H,
                        DType* x_ptr,
                        DType* hx_ptr,
                        DType* w_ptr,
                        DType* y_ptr,
                        DType* hy_ptr) {
  DType* wx = w_ptr;
  DType* wh = wx + I * H * 3;
  DType* bx = wh + H * H * 3 + (D - 1) * (H * H * 3 + I * H * 3)
      + (L - 1) * ((D + 1) * H) * H * 3 * D;
  DType* bh = bx + H * 3;
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(D * L, N, H));
  DType* hy_l = hy_ptr;
  DType* gateR_l = rs;
  DType* gateZ_l = gateR_l + L * T * D * N * H;
  DType* gateN_l = gateZ_l + L * T * D * N * H;
  DType* y_l = gateN_l + L * T * D * N * H;
  DType* Mnh_l = y_l + L * T * N * H * D;
  DType* tmp_buf = Mnh_l + L * D * T * N * H;
  DType* ws2 = Mnh_l + L * D * T * N * H + D * H * N;
  DType* wx_l = wx;
  DType* wh_l = wh;
  DType* bx_l = bx;
  DType* bh_l = bh;
  DType* y_tmp = x_ptr;

  for (int l = 0; l < L; l++) {
    if (l != 0) {
      y_tmp = y_l;
      y_l = y_l + T * N * H * D;
    }
    Tensor<cpu, 2, DType> x_l(y_tmp, Shape2(T * N, I));
    Tensor<cpu, 2, DType> hx_l = hx[D * l];
    GruForwardTrainingSingleLayer<DType>(ws2, tmp_buf, state_outputs, D, T, N, I, H,
                                         x_l, hx_l, wx_l, wh_l, bx_l, bh_l,
                                         gateR_l, gateZ_l, gateN_l, Mnh_l, y_l, hy_l);
    gateR_l = gateR_l + T * D * N * H;
    gateZ_l = gateZ_l + T * D * N * H;
    gateN_l = gateN_l + T * D * N * H;
    Mnh_l = Mnh_l +  T * D * N * H;
    hy_l = hy_l + D * N * H;
    bx_l = bx_l + 3 * H * D * 2;
    bh_l = bh_l + 3 * H * D * 2;

    wx_l = wx_l + I * H * 3 * D + H * H * 3 * D;
    if (l == 0) {
      I = D * H;
    }
    wh_l = wx_l + I * 3 * H;
  }
  #pragma omp parallel for
  for (int i = 0; i < T * N * H * D; i++) {
    y_ptr[i] = y_l[i];
  }
}

template <typename DType>
void GruBackwardSingleLayer(DType* ws,
                            DType* tmp_buf,
                            const int D,
                            const int T,
                            const int N,
                            const int I,
                            const int H,
                            const Tensor<cpu, 2, DType> &x,
                            const Tensor<cpu, 2, DType> &hx,
                            DType* wx_ptr,
                            DType* wh_ptr,
                            DType* y_ptr,
                            DType* dy_ptr,
                            DType* dhy_ptr,
                            DType* gateR,
                            DType* gateZ,
                            DType* gateN,
                            DType* Mnh,
                            DType* dx,
                            DType* dhx,
                            DType* dwx,
                            DType* dwh,
                            DType* dbx,
                            DType* dbh) {
  DType* dyt;
  DType* ht1;  // [N, D, H]
  DType* rt;
  DType* zt;
  DType* nt;
  DType* dat;
  DType* dart;
  DType* dar = ws;  // [T, N, 3 * H]
  DType* da = dar + T * N * 3 * H;  // [T, N, 3 * H]
  DType* dht1 = da + T * N * 3 * H;  // [D, N, H]
  DType* hx_ = dht1 + D * N * H;  // [N, D, H]
  DType* Mnht = Mnh;

  DType* back_ht1;
  DType* back_dht1 = dht1 + N * H;  // [N, H]
  DType* back_Mnht = Mnh + T * N * H;
  DType* back_gateR = gateR + T * N * H;
  DType* back_gateZ = gateZ + T * N * H;
  DType* back_gateN = gateN + T * N * H;
  DType* back_wx_ptr = wx_ptr + I * 3 * H + H * 3 * H;
  DType* back_wh_ptr = wh_ptr + I * 3 * H + H * 3 * H;
  DType* back_dwx = dwx + I * 3 * H + H * 3 * H;
  DType* back_dwh = dwh + I * 3 * H + H * 3 * H;
  DType* back_dbx = dbx + 3 * H * 2;
  DType* back_dbh = dbh + 3 * H * 2;

  DType alpha = 1.0;
  DType beta = 0.0;
  const Tensor<cpu, 2, DType> wx(wx_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh(wh_ptr, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> back_wx(back_wx_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> back_wh(back_wh_ptr, Shape2(H * 3, H));

  #pragma omp parallel for
  for (int i = 0; i < D * H * 3 * H; ++i) {
    dwh[i] = 0;
  }

  #pragma omp parallel for
  for (int i = 0; i < D * 3 * H; ++i) {
    dbx[i] = 0;
    dbh[i] = 0;
  }

  #pragma omp parallel for
  for (int i = 0; i < N * H; ++i) {
    if (dhy_ptr) {
      dht1[i] = dhy_ptr[i];
    } else {
      dht1[i] = 0;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) {
      hx_[i * D * H + j] = hx[i][j];
    }
  }

  if (D == BIDIRECT) {
    #pragma omp parallel for
    for (int i = 0; i < N * H; ++i) {
      if (dhy_ptr) {
        back_dht1[i] = dhy_ptr[N * H + i];
      } else {
        back_dht1[i] = 0;
      }
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        hx_[i * D * H + H + j] = hx[N + i][j];
      }
    }
  }
  for (int t = T - 1; t >= 0; --t) {
    if (t) {
      ht1 = y_ptr + (t - 1) * N * D * H;
    } else {
      ht1 = hx_;
    }
    // add dy[T, N, D, H] to dhy[D, N, H]
    dyt = dy_ptr + t * N * D * H;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        dht1[i * H + j] += dyt[i * D * H + j];
      }
    }

    rt = gateR + t * N * H;
    zt = gateZ + t * N * H;
    nt = gateN + t * N * H;
    Mnht = Mnh +  t * N * H;
    dat = da + t * N * 3 * H;
    dart = dar + t * N * 3 * H;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        int nid = i * 3 * H + 2 * H + j;
        int zid = i * 3 * H + H + j;
        int rid = i * 3 * H + j;
        int id = i * H + j;
        dat[nid] = dht1[id] * (1 - zt[id]) * (1 - nt[id] * nt[id]);
        dart[zid] = dat[zid] = dht1[id] * (ht1[i * D * H + j] - nt[id]) *
            zt[id] * (1 - zt[id]);
        dart[rid] = dat[rid] = dat[nid] * Mnht[id] * rt[id] *
            (1 - rt[id]);
        dart[nid] = dat[nid] * rt[id];
        dht1[id] = dht1[id] * zt[id];
      }
    }
    alpha = 1.0;
    beta = 1.0;

    // dht1 = dart * wh    [N, H] = [N, 3 * H] * [3 * H, H]
    Tensor<cpu, 2, DType> d_dht1(dht1, Shape2(N, H));
    Tensor<cpu, 2, DType> d_dart(dart, Shape2(N, 3 * H));
    linalg_gemm(d_dart, wh, d_dht1, alpha, beta, false, false);

    // dwh = dart.T * ht1    [3 * H, H] = [3 * H, N] * [N, H]
    Tensor<cpu, 2, DType> d_ht1(ht1, Shape2(N, D * H));
    Tensor<cpu, 2, DType> d_dwh(dwh, Shape2(3 * H, H));
    Tensor<cpu, 3, DType> d_ht1_tmp = Tensor<cpu, 3, DType>
        (reinterpret_cast<DType*>(tmp_buf), Shape3(D, H, N));
    d_ht1_tmp = reshape(d_ht1.T(), Shape3(D, H, N));
    linalg_gemm(d_dart, d_ht1_tmp[0], d_dwh, alpha, beta, true, true);
  }

  // dbx = e * da       [1, 3 * H] = [1, N] * [N, 3 * H]
  #pragma omp parallel for
  for (int i = 0; i < 3 * H; ++i) {
    for (int j = 0; j < N * T; ++j) {
      dbx[i] += da[j * 3 * H + i];
      dbh[i] += dar[j * 3 * H + i];
    }
  }
  alpha = 1.0;
  beta = 0.0;

  // dx = da * wx    [T * N, I] = [T * N, 3 * H] * [3 * H, I]
  Tensor<cpu, 2, DType> d_da(da, Shape2(T * N, 3 * H));
  Tensor<cpu, 2, DType> d_dx(dx, Shape2(T * N, I));
  linalg_gemm(d_da, wx, d_dx, alpha, beta, false, false);

  // dwx = da.T * x    [3 * H, I] = [3 * H, T * N] * [T * N, I]
  Tensor<cpu, 2, DType> d_dwx(dwx, Shape2(3 * H, I));
  linalg_gemm(d_da, x, d_dwx, alpha, beta, true, false);

  if (D == BIDIRECT) {
    for (int t = 0; t < T; ++t) {
      if (t == T-1) {
        back_ht1 = hx_;
      } else {
        back_ht1 = y_ptr + (t + 1) * N * D * H;
      }

      //  add dy[T, N, D, H] to dhy[D, N, H]
      dyt = dy_ptr + t * N * D * H;
      #pragma omp parallel for
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < H; ++j) {
          back_dht1[i * H + j] += dyt[i * D * H + H + j];
        }
      }

      rt = back_gateR + t * N * H;
      zt = back_gateZ + t * N * H;
      nt = back_gateN + t * N * H;
      back_Mnht = Mnh + (T + t) * N * H;
      dat = da + t * N * 3 * H;
      dart = dar + t * N * 3 * H;

      #pragma omp parallel for
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < H; ++j) {
          int nid = i * 3 * H + 2 * H + j;
          int zid = i * 3 * H + H + j;
          int rid = i * 3 * H + j;
          int id = i * H + j;
          dat[nid] = back_dht1[id] * (1 - zt[id]) * (1 - nt[id] * nt[id]);
          dart[zid] = dat[zid] = back_dht1[id] * (back_ht1[i * D * H + H + j] -
              nt[id]) * zt[id] * (1 - zt[id]);
          dart[rid] = dat[rid] = dat[nid] * back_Mnht[id] * rt[id] *
              (1 - rt[id]);
          dart[nid] = dat[nid] * rt[id];
          back_dht1[id] = back_dht1[id] * zt[id];
        }
      }
      alpha = 1.0;
      beta = 1.0;
      // dht1 = da * wh    [N, H] = [N, 3 * H] * [3 * H, H]
      Tensor<cpu, 2, DType> d_dart(dart, Shape2(N, 3 * H));
      Tensor<cpu, 2, DType> d_back_dht1(back_dht1, Shape2(N, H));
      linalg_gemm(d_dart, back_wh, d_back_dht1, alpha, beta, false, false);

      // dwh = da.T * ht1     [3 * H, H] = [3 * H, N] * [N, H]
      Tensor<cpu, 2, DType> d_back_dwh(back_dwh, Shape2(3 * H, H));
      Tensor<cpu, 2, DType> d_back_ht1(back_ht1 + H, Shape2(N, D * H));
      Tensor<cpu, 3, DType> d_back_ht1_tmp = Tensor<cpu, 3, DType>
          (reinterpret_cast<DType*>(tmp_buf), Shape3(D, H, N));
      d_back_ht1_tmp = reshape(d_back_ht1.T(), Shape3(D, H, N));
      linalg_gemm(d_dart, d_back_ht1_tmp[0], d_back_dwh, alpha, beta, true, true);
    }

    // dbx = e * da       [1, 3 * H] = [1, N] * [N, 3 * H]
    #pragma omp parallel for
    for (int i = 0; i < 3 * H; ++i) {
      for (int j = 0; j < N * T; ++j) {
        back_dbx[i] += da[j * 3 * H + i];
        back_dbh[i] += dar[j * 3 * H + i];
      }
    }
    alpha = 1.0;
    beta = 1.0;
    // dxt = da * wx    [T * N, I] = [T * N, 3 * H] * [3 * H, I]
    Tensor<cpu, 2, DType> d_da2(da, Shape2(T * N, 3 * H));
    Tensor<cpu, 2, DType> d_dx(dx, Shape2(T * N, I));
    linalg_gemm(d_da2, back_wx, d_dx, alpha, beta, false, false);
    alpha = 1.0;
    beta = 0.0;
    // dwx = da.T * xt    [3 * H, I] = [3 * H, N] * [N, I]
    Tensor<cpu, 2, DType> d_back_dwx(back_dwx, Shape2(3 * H, I));
    linalg_gemm(d_da2, x, d_back_dwx, alpha, beta, true, false);
  }
  #pragma omp parallel for
  for (int i = 0; i < D * N * H; ++i) {
    dhx[i] = dht1[i];
  }
}

template <typename DType>
void GruBackward(DType* ws,
                 DType* rs,
                 const int L,
                 const int D,
                 const int T,
                 const int N,
                 int I,
                 const int H,
                 DType* x_ptr,
                 DType* hx_ptr,
                 DType* w_ptr,
                 DType* dy_ptr,
                 DType* dhy_ptr,
                 DType* dx_ptr,
                 DType* dhx_ptr,
                 DType* dw_ptr) {
  DType* wx = w_ptr;
  DType* dwx = dw_ptr;
  DType* dwh = dwx + I * H * 3;
  DType* dbx = dwh + H * H * 3 + (D - 1) * (H * H * 3 + I * H * 3)
      + (L - 1) * ((D + 1) * H) * H * 3 * D;
  DType* gateR_l = rs + (L - 1) * T * D * N * H;
  DType* gateZ_l = gateR_l + L * T * D * N * H;
  DType* gateN_l = gateZ_l + L * T * D * N * H;
  DType* y_l = gateN_l + L * T * D * N * H;
  DType* Mnh_l = y_l + L * T * N * H * D;
  DType* tmp_buf = Mnh_l + L * D * T * N * H;
  DType* dx_l = tmp_buf + T * N * D * H;
  DType* ws2 = Mnh_l + L * T * N * H * D + T * N * D * H + T * N * D * H;
  DType* wx_l = (L == 1)? wx : wx + (L - 2) * D * (D + 1) * H * 3 * H
      + D * I * 3 * H + D * H * 3 * H;
  DType* wh_l = wx_l;
  if (L == 1) {
    wh_l = wh_l + I * H * 3;
  } else {
    wh_l = wh_l + (D * H) * H * 3;
  }
  DType* dhy_l = NULL;
  if (dhy_ptr)
    dhy_l = dhy_ptr + (L - 1) * D * N * H;
  DType* dwx_l = (L == 1)? dwx : dwx + (L - 2) * D * (D + 1) * H * 3 * H
      + D * I * 3 * H + D * H * 3 * H;
  DType* dwh_l = NULL;
  if (L == 1) {
    dwh_l = dwx_l + I * H * 3;
  } else {
    dwh_l = dwx_l + (D * H) * H * 3;
  }
  DType* dbx_l = dbx + (L - 1) * D * 3 * H * 2;
  DType* dbh_l = dbx_l + 3 * H;
  DType* dhx_l = dhx_ptr + (L - 1) * D * N * H;
  DType* dy_l = dy_ptr;
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, D * N, H));
  int inputsize = I;
  DType* y_tmp = y_l - T * N * H * D;
  for (int l = L - 1; l >= 0; --l) {
    if (l == 0) {
      I = inputsize;
      y_tmp = x_ptr;
      dx_l = dx_ptr;
    } else {
      I = D * H;
    }
    Tensor<cpu, 2, DType> hx_l = hx[l];
    Tensor<cpu, 2, DType> x_l(y_tmp, Shape2(T * N, I));
    GruBackwardSingleLayer<DType>(ws2, tmp_buf, D, T, N, I, H, x_l, hx_l, wx_l, wh_l, y_l, dy_l,
                                  dhy_l, gateR_l, gateZ_l, gateN_l, Mnh_l, dx_l, dhx_l,
                                  dwx_l, dwh_l, dbx_l, dbh_l);
    if (l > 0) {
      #pragma omp parallel for
      for (int i = 0; i < T * N * D * H; ++i) {
        dy_l[i] = dx_l[i];
      }
      gateR_l = gateR_l - T * D * N * H;
      gateZ_l = gateZ_l - T * D * N * H;
      gateN_l = gateN_l - T * D * N * H;
      Mnh_l = Mnh_l -  T * D * N * H;
      dhx_l = dhx_l - D * N * H;
      if (dhy_l)
        dhy_l = dhy_l - D * N * H;
      y_l = y_l - T * N * H * D;
      y_tmp = y_l;
      if (l == 1) {
        wx_l = wx_l - (inputsize + H) * H * 3 * D;
        wh_l = wx_l + inputsize * 3 * H;
        dwx_l = dwx_l - (inputsize + H) * H * 3 * D;
        dwh_l = dwx_l + inputsize * 3 * H;
      } else {
        wx_l = wx_l - (I + H) * H * 3 * D;
        wh_l = wx_l + I * 3 * H;
        dwx_l = dwx_l - (I + H) * H * 3 * D;
        dwh_l = dwx_l + I * 3 * H;
      }
      dbx_l = dbx_l - D * 3 * H * 2;
      dbh_l = dbx_l + 3 * H;
    }
  }
}
#endif  // MXNET_OPERATOR_RNN_IMPL_HPP_
