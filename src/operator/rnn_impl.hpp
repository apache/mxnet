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
void LstmForwardTrainingSingleLayer(DType* ws,
                                    DType* rs,
                                    const int D,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    const Tensor<cpu, 2, DType> &x,
                                    const Tensor<cpu, 2, DType> &hx,
                                    const Tensor<cpu, 2, DType> &cx,
                                    DType* w_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, 4 * H));
  const Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, 4 * H));
  const Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
  const Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
  Tensor<cpu, 3, DType> h(rs, Shape3(T, N, H));
  Tensor<cpu, 3, DType> c(rs + T * N * H, Shape3(T, N, H));
  Tensor<cpu, 4, DType> ifgo(rs + T * N * H * 2, Shape4(T, N, H, 4));

  const DType alpha = 1.0;
  const DType beta = 0.0;
  linalg_gemm(x, wx, yx_flat, alpha, beta, false, true);

  for (int i = 0; i < T; ++i) {
    linalg_gemm((i == 0) ? hx : h[i-1], wh, yh_flat, alpha, beta, false, true);
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < H; ++k) {
        DType it = sigmoid<DType>(yx[i][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
        DType ft = sigmoid<DType>(yx[i][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
        DType gt =           tanh(yx[i][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
        DType ot = sigmoid<DType>(yx[i][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
        DType ct = ((i == 0) ? cx[j][k] : c[i-1][j][k]) * ft + it * gt;
        h[i][j][k] = ot * tanh(ct);
        c[i][j][k] = ct;
        // reserve
        ifgo[i][j][k][0] = it;
        ifgo[i][j][k][1] = ft;
        ifgo[i][j][k][2] = gt;
        ifgo[i][j][k][3] = ot;
      }
    }
  }
}

template <typename DType>
void LstmForwardTraining(DType* ws,
                         DType* rs,
                         bool state_outputs,
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
                         DType* y_ptr,
                         DType* hy_ptr,
                         DType* cy_ptr) {
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(L, N, H));
  LstmForwardTrainingSingleLayer<DType>(ws, rs, D, T, N, I, H, x, hx[0], cx[0], w_ptr);
  if (state_outputs) {
    memcpy(hy_ptr, rs + (T - 1) * N * H, N * H * sizeof(DType));
    memcpy(cy_ptr, rs + (T + T - 1) * N * H, N * H * sizeof(DType));
  }
  memcpy(y_ptr, rs, T * N * H * sizeof(DType));
}

template<typename DType>
void LstmForwardInferenceSingleLayer(DType* ws,
                                     bool state_outputs,
                                     const int D,
                                     const int T,
                                     const int N,
                                     const int I,
                                     const int H,
                                     const Tensor<cpu, 2, DType> &x,
                                     const Tensor<cpu, 2, DType> &hx,
                                     const Tensor<cpu, 2, DType> &cx,
                                     DType* w_ptr,
                                     DType* y_ptr,
                                     DType* hy_ptr,
                                     DType* cy_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  const Tensor<cpu, 2, DType> bx(wh.dptr_ + H * H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh(bx.dptr_ + H * 4, Shape2(4, H));
  Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, H * 4));
  Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, H * 4));
  const Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
  const Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
  Tensor<cpu, 2, DType> c(yh_flat.dptr_ + N * H * 4, Shape2(N, H));
  Tensor<cpu, 3, DType> h(y_ptr, Shape3(T, N, H));
  const DType alpha = 1.0;
  const DType beta = 0.0;
  linalg_gemm(x, wx, yx_flat, alpha, beta, false, true);

  for (int i = 0; i < T; ++i) {
    linalg_gemm((i == 0) ? hx : h[i-1], wh, yh_flat, alpha, beta, false, true);
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < H; ++k) {
        DType it = sigmoid<DType>(yx[i][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
        DType ft = sigmoid<DType>(yx[i][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
        DType gt =           tanh(yx[i][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
        DType ot = sigmoid<DType>(yx[i][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
        DType ct = ((i == 0) ? cx[j][k] : c[j][k]) * ft + it * gt;
        h[i][j][k] = ot * tanh(ct);
        c[j][k] = ct;
      }
    }
  }
  if (state_outputs) {
    memcpy(hy_ptr, y_ptr + (T - 1) * N * H, N * H * sizeof(DType));
    memcpy(cy_ptr, c.dptr_, N * H * sizeof(DType));
  }
}

template <typename DType>
void LstmForwardInference(DType* ws,
                          bool state_outputs,
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
                          DType* y_ptr,
                          DType* hy_ptr,
                          DType* cy_ptr) {
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(L, N, H));
  LstmForwardInferenceSingleLayer<DType>(ws, state_outputs, D, T, N, I, H,
                                         x, hx[0], cx[0], w_ptr, y_ptr, hy_ptr, cy_ptr);
}

template <typename DType>
void LstmBackwardSingleLayer(DType* ws,
                             DType* rs,
                             const int D,
                             const int T,
                             const int N,
                             const int I,
                             const int H,
                             const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &hx,
                             const Tensor<cpu, 2, DType> &cx,
                             const Tensor<cpu, 3, DType> &y,
                             const Tensor<cpu, 3, DType> &dy,
                             const Tensor<cpu, 2, DType> &dx,
                             const Tensor<cpu, 2, DType> &dhx,
                             const Tensor<cpu, 2, DType> &dcx,
                             DType* dhy_ptr,
                             DType* dcy_ptr,
                             DType* w_ptr,
                             DType* dw_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  Tensor<cpu, 2, DType> dwx(dw_ptr, Shape2(H * 4, I));
  Tensor<cpu, 2, DType> dwh(dw_ptr + I * H * 4, Shape2(H * 4, H));
  Tensor<cpu, 1, DType> dbx(dwh.dptr_ + H * H * 4, Shape1(H * 4));
  Tensor<cpu, 1, DType> dbh(dbx.dptr_ + H * 4, Shape1(H * 4));
  const Tensor<cpu, 3, DType> h(rs, Shape3(T, N, H));
  const Tensor<cpu, 3, DType> c(rs + T * N * H, Shape3(T, N, H));
  const Tensor<cpu, 4, DType> ifgo(rs + T * N * H * 2, Shape4(T, N, H, 4));

  memset(dwh.dptr_, 0, H * H * 4 * sizeof(DType));
  memset(dbx.dptr_, 0, H * 4 * sizeof(DType));
  memset(dbh.dptr_, 0, H * 4 * sizeof(DType));
  Tensor<cpu, 4, DType> difgo(ws, Shape4(T, N, 4, H));
  Tensor<cpu, 2, DType> dh(ws + T * N * H * 4, Shape2(N, H));
  Tensor<cpu, 2, DType> dc(dh.dptr_ + N * H, Shape2(N, H));
  if (dhy_ptr != NULL) {
    memcpy(dh.dptr_, dhy_ptr, N * H * sizeof(DType));
  }
  if (dcy_ptr != NULL) {
    memcpy(dc.dptr_, dcy_ptr, N * H * sizeof(DType));
  }
  const DType alpha = 1.0;
  const DType beta0 = 0.0;
  const DType beta1 = 1.0;
  for (int i = T - 1; i >= 0; --i) {
    const Tensor<cpu, 2, DType>& dhnext = i ? dh : dhx;
    const Tensor<cpu, 2, DType>& dcnext = i ? dc : dcx;
    const Tensor<cpu, 2, DType>& hnext = i ? h[i-1] : hx;
    const Tensor<cpu, 2, DType>& cnext = i ? c[i-1] : cx;
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < H; ++k) {
         DType tc = tanh(c[i][j][k]);
         DType it = ifgo[i][j][k][0];
         DType ft = ifgo[i][j][k][1];
         DType gt = ifgo[i][j][k][2];
         DType ot = ifgo[i][j][k][3];

         dh[j][k] += dy[i][j][k];
         dc[j][k] += dh[j][k] * ot * (1 - tc * tc);

         difgo[i][j][0][k] = dc[j][k] * gt * it * (1 - it);
         difgo[i][j][1][k] = dc[j][k] * cnext[j][k] * ft * (1 - ft);
         difgo[i][j][2][k] = dc[j][k] * it * (1 - gt * gt);
         difgo[i][j][3][k] = dh[j][k] * tc * ot * (1 - ot);
         dcnext[j][k] = dc[j][k] * ft;
      }
    }
    Tensor<cpu, 2, DType> dyh(difgo[i].dptr_, Shape2(N, H * 4));
    linalg_gemm(dyh, wh, dhnext, alpha, beta0, false, false);
    linalg_gemm(dyh, hnext, dwh, alpha, beta1, true, false);
  }
  Tensor<cpu, 2, DType> dyx(difgo.dptr_, Shape2(T * N, H * 4));
  linalg_gemm(dyx, wx, dx, alpha, beta0, false, false);
  linalg_gemm(dyx, x, dwx, alpha, beta0, true, false);
  for (int i = 0; i < T * N; ++i) {
    for (int j = 0; j < H * 4; ++j) {
      dbx[j] += dyx[i][j];
      dbh[j] = dbx[j];
    }
  }
}

template <typename DType>
void LstmBackward(DType* ws,
                  DType* rs,
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
                  DType* y_ptr,
                  DType* dy_ptr,
                  DType* dhy_ptr,
                  DType* dcy_ptr,
                  DType* dx_ptr,
                  DType* dhx_ptr,
                  DType* dcx_ptr,
                  DType* dw_ptr) {
  Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(L, N, H));
  Tensor<cpu, 2, DType> dx(dx_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> dhx(dhx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> dcx(dcx_ptr, Shape3(L, N, H));
  Tensor<cpu, 3, DType> y(y_ptr, Shape3(T, N, H));
  Tensor<cpu, 3, DType> dy(dy_ptr, Shape3(T, N, H));

  // current layer dcx and dhx
  Tensor<cpu, 2, DType> dcx_cl(dcx[0].dptr_, Shape2(N, H));
  Tensor<cpu, 2, DType> dhx_cl(dhx[0].dptr_, Shape2(N, H));
  LstmBackwardSingleLayer<DType>(ws, rs, D, T, N, I, H, x, hx[0], cx[0], y, dy, dx,
                                 dhx_cl, dcx_cl, dhy_ptr, dcy_ptr, w_ptr, dw_ptr);
}
#endif  // MXNET_OPERATOR_RNN_IMPL_HPP_
