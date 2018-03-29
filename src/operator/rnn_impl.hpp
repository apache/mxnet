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

template<typename DType>
inline DType sigmoid(DType x) {
  return 1.0f / (1.0f + exp(-x));
}

template<typename DType>
void GruForwardInferenceSingleLayer(DType* ws,
                                    bool state_outputs,
                                    const int D,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    const Tensor<cpu, 2, DType> &x,
                                    const Tensor<cpu, 2, DType> &hx,
                                    const Tensor<cpu, 2, DType> &wx,
                                    const Tensor<cpu, 2, DType> &wh,
                                    const Tensor<cpu, 2, DType> &bx,
                                    const Tensor<cpu, 2, DType> &bh,
                                    DType* y_ptr,
                                    DType* hy_ptr) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
    for (int j = 0; j < H; j++) {
      y_ptr[i * H + j] = hx[i][j];
    }

  DType* ht = y_ptr;
  DType* ht_1 = y_ptr;
  DType* gemmC1  = ws;              // [D, T, N, 3 * H]
  DType* gemmC2  = gemmC1 + D * T * N * 3 * H;  // N * 3 * H
  DType* rt = gemmC2 + N * 3 * H;
  DType* zt = rt + N * H;
  DType* nt = zt + N * H;
  DType* gemmC1_t = gemmC1;
  Tensor<cpu, 2, DType> dgemmC1(ws, Shape2(D * T * N, 3 * H));
  Tensor<cpu, 2, DType> dgemmC2(gemmC2, Shape2(D * N, 3 * H));

  // x * wx.T : [T * N, I] * [I, 3 * H]
  DType alpha = 1.0;
  DType beta = 0.0;
  linalg_gemm(x, wx, dgemmC1, alpha, beta, false, true);

  for (int t = 0; t < T; t++) {
    //  perform the first direction, X * wx and H * wh for each step
    //  ht-1 * wh, ht-1:[N, H] wh:[3 * H, H]
    Tensor<cpu, 2, DType> dht_1(ht_1, Shape2(N, D * H));
    linalg_gemm(dht_1, wh, dgemmC2, alpha, beta, false, true);
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
  }
  //  copy last state to hy, from(N, H * D) to (D, N, H)
  if (state_outputs) {
    DType* y_start = y_ptr + (T - 1) * N * H;
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int j = 0; j < H; j++) {
        hy_ptr[i * H + j] = y_start[i * H + j];
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
                         const int I,
                         const int H,
                         DType* x_ptr,
                         DType* hx_ptr,
                         DType* w_ptr,
                         DType* y_ptr,
                         DType* hy_ptr) {
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 3, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 3, Shape2(3, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 3, Shape2(3, H));

  DType* y_tmp = ws;
  DType* y_l = x_ptr;
  DType* ws2 = y_tmp + D * T * N * H;

  const Tensor<cpu, 2, DType> wx_l = wx;
  const Tensor<cpu, 2, DType> wh_l = wh;
  const Tensor<cpu, 2, DType> bx_l = bx;
  const Tensor<cpu, 2, DType> bh_l = bh;
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> hy(hy_ptr, Shape3(L, N, H));
  Tensor<cpu, 2, DType> x_l = x;
  Tensor<cpu, 2, DType> hx_l = hx[0];
  DType* hy_l = hy_ptr;

  for (int i = 0; i < T * N; i++)
    for (int j = 0; j < I; j++) {
      x_l[i][j] = y_l[i * I + j];
    }

  y_l = y_ptr;

  GruForwardInferenceSingleLayer<DType>(ws2, state_outputs, D, T, N, I, H,
                                        x_l, hx_l, wx_l, wh_l, bx_l, bh_l, y_l, hy_l);
}


template<typename DType>
void GruForwardTrainingSingleLayer(DType* ws,
                                   bool state_outputs,
                                   const int D,
                                   const int T,
                                   const int N,
                                   const int I,
                                   const int H,
                                   const Tensor<cpu, 2, DType> &x,
                                   const Tensor<cpu, 2, DType> &hx,
                                   const Tensor<cpu, 2, DType> &wx,
                                   const Tensor<cpu, 2, DType> &wh,
                                   const Tensor<cpu, 2, DType> &bx,
                                   const Tensor<cpu, 2, DType> &bh,
                                   DType* gateR,
                                   DType* gateZ,
                                   DType* gateN,
                                   DType* Mnh,
                                   DType* y_ptr,
                                   DType* hy_ptr) {
  DType* ht = y_ptr;
  DType* ht_1 = y_ptr;
  DType* gemmC1  = ws;              // [D, T, N, 3 * H]
  DType* gemmC2  = gemmC1 + D * T * N * 3 * H;  // N * 3 * H
  DType* rt = gateR;
  DType* zt = gateZ;
  DType* nt = gateN;
  DType* gemmC1_t = gemmC1;
  Tensor<cpu, 2, DType> dgemmC1(ws, Shape2(D * T * N, 3 * H));
  Tensor<cpu, 2, DType> dgemmC2(gemmC2, Shape2(D * N, 3 * H));

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
    for (int j = 0; j < H; j++) {
      y_ptr[i * H + j] = hx[i][j];
    }

  // x * wx.T : [T * N, I] * [I, 3 * H]
  DType alpha = 1.0;
  DType beta = 0.0;
  linalg_gemm(x, wx, dgemmC1, alpha, beta, false, true);

  for (int t = 0; t < T; t++) {
    //  perform the first direction, X * wx and H * wh for each step
    //  ht-1 * wh, ht-1:[N, H] wh:[3 * H, H]

    Tensor<cpu, 2, DType> dht_1(ht_1, Shape2(N, D * H));
    linalg_gemm(dht_1, wh, dgemmC2, alpha, beta, false, true);
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
  }
  //  copy last state to hy, from(N, H * D) to (D, N, H)
  if (state_outputs) {
    DType* y_start = y_ptr + (T - 1) * N * H;
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
      for (int j = 0; j < H; j++) {
        hy_ptr[i * H + j] = y_start[i * H + j];
      }
  }
}

template <typename DType>
void GruForwardTraining(DType* ws,
                        bool state_outputs,
                        const int L,
                        const int D,
                        const int T,
                        const int N,
                        const int I,
                        const int H,
                        DType* x_ptr,
                        DType* hx_ptr,
                        DType* w_ptr,
                        DType* y_ptr,
                        DType* hy_ptr) {
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 3, Shape2(H * 3, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 3, Shape2(3, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 3, Shape2(3, H));
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> hy(hy_ptr, Shape3(L, N, H));
  Tensor<cpu, 2, DType> x_l = x;
  Tensor<cpu, 2, DType> hx_l = hx[0];
  DType* hy_l = hy_ptr;
  DType* gateR_l = ws;
  DType* gateZ_l = gateR_l + L * T * D * N * H;
  DType* gateN_l = gateZ_l + L * T * D * N * H;
  DType* y_l = gateN_l + L * T * D * N * H;
  DType* Mnh_l = y_l + L * T * N * H * D;
  DType* ws2 = Mnh_l + L * D * T * N * H;
  const Tensor<cpu, 2, DType> wx_l = wx;
  const Tensor<cpu, 2, DType> wh_l = wh;
  const Tensor<cpu, 2, DType> bx_l = bx;
  const Tensor<cpu, 2, DType> bh_l = bh;

  GruForwardTrainingSingleLayer<DType>(ws2, state_outputs, D, T, N, I, H,
                                       x_l, hx_l, wx_l, wh_l, bx_l, bh_l,
                                       gateR_l, gateZ_l, gateN_l, Mnh_l, y_l, hy_l);

  #pragma omp parallel for
  for (int i = 0; i < T * N * H * D; i++) {
    y_ptr[i] = y_l[i];
  }
}

template <typename DType>
void GruBackwardSingleLayer(DType* ws,
                            const int D,
                            const int T,
                            const int N,
                            const int I,
                            const int H,
                            const Tensor<cpu, 2, DType> &x,
                            const Tensor<cpu, 2, DType> &hx,
                            const Tensor<cpu, 2, DType> &wx,
                            const Tensor<cpu, 2, DType> &wh,
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
  DType alpha = 1.0;
  DType beta = 0.0;

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
    dht1[i] = dhy_ptr[i];
  }

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) {
      hx_[i * D * H + j] = hx[i][j];
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
    Tensor<cpu, 2, DType> d_ht1(ht1, Shape2(N, H));
    Tensor<cpu, 2, DType> d_dwh(dwh, Shape2(3 * H, H));
    linalg_gemm(d_dart, d_ht1, d_dwh, alpha, beta, true, false);
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
  // dx = da * wx    [T * N, I] = [T * N,3 * H] * [3 * H, I]
  Tensor<cpu, 2, DType> d_da(da, Shape2(T * N, 3 * H));
  Tensor<cpu, 2, DType> d_dx(dx, Shape2(T * N, I));
  linalg_gemm(d_da, wx, d_dx, alpha, beta, false, false);

  // dwx = da.T * x    [3 * H, I] = [3 * H, T * N] * [T * N, I]
  Tensor<cpu, 2, DType> d_dwx(dwx, Shape2(3 * H, I));
  linalg_gemm(d_da, x, d_dwx, alpha, beta, true, false);

  #pragma omp parallel for
  for (int i = 0; i < D * N * H; ++i) {
    dhx[i] = dht1[i];
  }
}

template <typename DType>
void GruBackward(DType* ws,
                 const int L,
                 const int D,
                 const int T,
                 const int N,
                 const int I,
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
  DType* wh = wx + I * H * 3 * D;
  DType* dwx = dw_ptr;
  DType* dwh = dwx + I * H * 3 * D;
  DType* dbx = dwh + H * H * 3 * D;
  DType* dbh = dbx + H * 3 * D;
  DType* gateR_l = ws + (L - 1) * T * D * N * H;
  DType* gateZ_l = gateR_l + L * T * D * N * H;
  DType* gateN_l = gateZ_l + L * T * D * N * H;
  DType* y_l = gateN_l + L * T * D * N * H;
  DType* Mnh_l = y_l + L * T * N * H * D;
  DType* ws2 = Mnh_l + T * N * H * D;
  DType* wx_l_ptr = (L == 1)? wx : wx + (L - 2) * D * (D * H) * 3 * H + D * I * 3 * H;
  DType* wh_l_ptr = wh + (L - 1) * D * H * 3 * H;
  DType* x_l_ptr = x_ptr;
  DType* hx_l_ptr = hx_ptr + (L - 1) * D * N * H;
  DType* dhy_l = dhy_ptr + (L - 1) * D * N * H;
  DType* dwx_l = (L == 1)? dwx : dwx + (L - 2) * D * (D * H) * 3 * H + D * I * 3 * H;
  DType* dwh_l = dwh + (L - 1) * D * H * 3 * H;
  DType* dbx_l = dbx + (L - 1) * D * 3 * H;
  DType* dbh_l = dbh + (L - 1) * D * 3 * H;
  DType* dx_l = dx_ptr;
  DType* dhx_l = dhx_ptr + (L - 1) * D * N * H;
  DType* dy_l = dy_ptr;
  const Tensor<cpu, 2, DType> wx_l(wx_l_ptr, Shape2(H * 3, I));
  const Tensor<cpu, 2, DType> wh_l(wh_l_ptr, Shape2(H * 3, H));
  Tensor<cpu, 2, DType> x_l(x_l_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_l_ptr, Shape3(L, N, H));
  Tensor<cpu, 2, DType> hx_l = hx[0];

  GruBackwardSingleLayer<DType>(ws2, D, T, N, I, H, x_l, hx_l, wx_l, wh_l, y_l, dy_l,
                                dhy_l, gateR_l, gateZ_l, gateN_l, Mnh_l, dx_l, dhx_l,
                                dwx_l, dwh_l, dbx_l, dbh_l);
}
#endif  // MXNET_OPERATOR_RNN_IMPL_HPP_
