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
 * \file    rnn_impl.h
 * \brief
 * \author  Shu Zhang
*/
#ifndef MXNET_OPERATOR_RNN_IMPL_H_
#define MXNET_OPERATOR_RNN_IMPL_H_

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
                                    bool state_outputs,
                                    bool bid,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    const Tensor<cpu, 2, DType> &x,
                                    const Tensor<cpu, 2, DType> &hx,
                                    const Tensor<cpu, 2, DType> &cx,
                                    const Tensor<cpu, 3, DType> &y,
                                    DType* w_ptr,
                                    DType* b_ptr,
                                    DType* hy_ptr,
                                    DType* cy_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  const Tensor<cpu, 2, DType> bx(b_ptr, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh(b_ptr + H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, 4 * H));
  const Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, 4 * H));
  const Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
  const Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
  Tensor<cpu, 2, DType> h(yh_flat.dptr_ + N * H * 4, Shape2(N, H));
  DType *c_ptr = bid ? rs + T * N * H * 7 : rs;
  Tensor<cpu, 3, DType> c(c_ptr, Shape3(T, N, H));
  Tensor<cpu, 4, DType> ifgo(c_ptr + T * N * H, Shape4(T, N, H, 4));

  const int offset = bid ? H : 0;
  const DType alpha = 1.0;
  const DType beta = 0.0;
  const int cell_size = N * H;
  linalg_gemm(x, wx, yx_flat, alpha, beta, false, true);

  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  for (int i = 0; i < T; ++i) {
    int t = bid ? T - 1 - i : i;
    linalg_gemm(i ? h : hx, wh, yh_flat, alpha, beta, false, true);
    #pragma omp parallel for num_threads(omp_threads)
    for (int jk = 0; jk < cell_size; ++jk) {
      int j = jk / H;
      int k = jk % H;
      DType it = sigmoid<DType>(yx[t][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
      DType ft = sigmoid<DType>(yx[t][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
      DType gt =           tanh(yx[t][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
      DType ot = sigmoid<DType>(yx[t][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
      DType ct = (i ? c[i-1][j][k] : cx[j][k]) * ft + it * gt;
      DType ht = ot * tanh(ct);
      h[j][k] = ht;
      // reserve
      y[t][j][k + offset] = ht;
      c[i][j][k] = ct;
      ifgo[i][j][k][0] = it;
      ifgo[i][j][k][1] = ft;
      ifgo[i][j][k][2] = gt;
      ifgo[i][j][k][3] = ot;
      if (i == T - 1 && state_outputs) {
        hy_ptr[jk] = ht;
        cy_ptr[jk] = ct;
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
                         DType* b_ptr,
                         DType* y_ptr,
                         DType* hy_ptr,
                         DType* cy_ptr) {
  const int total_layers = D * L;
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(total_layers, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(total_layers, N, H));
  const int b_size = 2 * H * 4;
  const int r_size = D * T * N * H * 6;
  const int y_offset = T * N * H * 5;
  const int cell_size = N * H;
  int idx = 0;  // state & cell state's idx;
  for (int i = 0; i < L; ++i) {
    const int input_size = i ? H * D : I;
    const int w_size = (input_size + H) * H * 4;
    Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, input_size));
    Tensor<cpu, 3, DType> y(rs + y_offset, Shape3(T, N, H * D));
    LstmForwardTrainingSingleLayer<DType>(ws, rs, state_outputs, false, T, N, input_size, H, x,
                                          hx[idx], cx[idx], y, w_ptr, b_ptr, hy_ptr, cy_ptr);
    if (D == 2) {
      w_ptr += w_size;
      b_ptr += b_size;
      ++idx;
      if (state_outputs) {
        hy_ptr += cell_size;
        cy_ptr += cell_size;
      }
      LstmForwardTrainingSingleLayer<DType>(ws, rs, state_outputs, true, T, N, input_size, H, x,
                                            hx[idx], cx[idx], y, w_ptr, b_ptr, hy_ptr, cy_ptr);
    }
    if (i != L - 1) {
      w_ptr += w_size;
      b_ptr += b_size;
      x_ptr = y.dptr_;
      rs += r_size;
      ++idx;
      if (state_outputs) {
        hy_ptr += cell_size;
        cy_ptr += cell_size;
      }
    }
  }
  memcpy(y_ptr, rs + y_offset, T * N * H * D * sizeof(DType));
}

template<typename DType>
void LstmForwardInferenceSingleLayer(DType* ws,
                                     bool state_outputs,
                                     bool bid,
                                     const int T,
                                     const int N,
                                     const int I,
                                     const int H,
                                     const Tensor<cpu, 2, DType> &x,
                                     const Tensor<cpu, 2, DType> &hx,
                                     const Tensor<cpu, 2, DType> &cx,
                                     const Tensor<cpu, 3, DType> &y,
                                     DType* w_ptr,
                                     DType* b_ptr,
                                     DType* hy_ptr,
                                     DType* cy_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  const Tensor<cpu, 2, DType> bx(b_ptr, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh(b_ptr + H * 4, Shape2(4, H));
  Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, H * 4));
  Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, H * 4));
  const Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
  const Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
  Tensor<cpu, 2, DType> h(yh_flat.dptr_ + N * H * 4, Shape2(N, H));
  Tensor<cpu, 2, DType> c(h.dptr_ + N * H, Shape2(N, H));
  const int offset = bid ? H : 0;
  const DType alpha = 1.0;
  const DType beta = 0.0;
  const int cell_size = N * H;
  linalg_gemm(x, wx, yx_flat, alpha, beta, false, true);

  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  for (int i = 0; i < T; ++i) {
    int t = bid ? T - 1 - i : i;
    linalg_gemm(i ? h : hx, wh, yh_flat, alpha, beta, false, true);
    #pragma omp parallel for num_threads(omp_threads)
    for (int jk = 0; jk < cell_size; ++jk) {
      int j = jk / H;
      int k = jk % H;
      DType it = sigmoid<DType>(yx[t][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
      DType ft = sigmoid<DType>(yx[t][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
      DType gt =           tanh(yx[t][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
      DType ot = sigmoid<DType>(yx[t][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
      DType ct = (i ? c[j][k] : cx[j][k]) * ft + it * gt;
      DType ht = ot * tanh(ct);
      y[t][j][k + offset] = ht;
      if (i == T - 1 && state_outputs) {
        hy_ptr[jk] = ht;
        cy_ptr[jk] = ct;
      } else {
        h[j][k] = ht;
        c[j][k] = ct;
      }
    }
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
                          DType* b_ptr,
                          DType* y_ptr,
                          DType* hy_ptr,
                          DType* cy_ptr) {
  const int total_layers = D * L;
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(total_layers, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(total_layers, N, H));
  const int b_size = 2 * H * 4;
  const int cell_size = N * H;
  DType* y_tmp_ptr = ws + (T + 1) * cell_size * 4 + cell_size * 2;
  DType* y_cur_ptr = y_ptr;
  int idx = 0;  // state & cell state's idx;
  bool flag = L % 2 ? false : true;
  for (int i = 0; i < L; ++i) {
    const int input_size = i ? H * D : I;
    const int w_size = (input_size + H) * H * 4;
    // If bidirectional, need space to save current layer output y.
    if (D == 2) {
      y_cur_ptr = flag ? y_tmp_ptr : y_ptr;
      flag = !flag;
    }
    Tensor<cpu, 2, DType> x(x_ptr, Shape2(T * N, input_size));
    Tensor<cpu, 3, DType> y(y_cur_ptr, Shape3(T, N, H * D));
    LstmForwardInferenceSingleLayer<DType>(ws, state_outputs, false, T, N, input_size, H,
                                           x, hx[idx], cx[idx], y, w_ptr, b_ptr, hy_ptr, cy_ptr);
    // If bidirectional, then calculate the reverse direction's forward result.
    if (D == 2) {
      w_ptr += w_size;
      b_ptr += b_size;
      ++idx;
      if (state_outputs) {
        hy_ptr += cell_size;
        cy_ptr += cell_size;
      }
      LstmForwardInferenceSingleLayer<DType>(ws, state_outputs, true, T, N, input_size, H,
                                             x, hx[idx], cx[idx], y, w_ptr, b_ptr, hy_ptr, cy_ptr);
    }
    // Don't need to move pointer in the last layer.
    if (i != L - 1) {
      w_ptr += w_size;
      b_ptr += b_size;
      x_ptr = y_cur_ptr;
      ++idx;
      if (state_outputs) {
        hy_ptr += cell_size;
        cy_ptr += cell_size;
      }
    }
  }
}

template <typename DType>
void LstmBackwardSingleLayer(DType* ws,
                             DType* rs,
                             bool bid,
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
                             DType* dw_ptr,
                             DType* db_ptr) {
  using namespace mshadow;
  const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(H * 4, I));
  const Tensor<cpu, 2, DType> wh(w_ptr + I * H * 4, Shape2(H * 4, H));
  Tensor<cpu, 2, DType> dwx(dw_ptr, Shape2(H * 4, I));
  Tensor<cpu, 2, DType> dwh(dw_ptr + I * H * 4, Shape2(H * 4, H));
  Tensor<cpu, 1, DType> dbx(db_ptr, Shape1(H * 4));
  Tensor<cpu, 1, DType> dbh(dbx.dptr_ + H * 4, Shape1(H * 4));
  DType *c_ptr = bid ? rs + T * N * H * 7 : rs;
  const Tensor<cpu, 3, DType> c(c_ptr, Shape3(T, N, H));
  const Tensor<cpu, 4, DType> ifgo(c_ptr + T * N * H, Shape4(T, N, H, 4));
  memset(dwh.dptr_, 0, H * H * 4 * sizeof(DType));
  memset(dbx.dptr_, 0, H * 4 * sizeof(DType));
  memset(dbh.dptr_, 0, H * 4 * sizeof(DType));
  Tensor<cpu, 4, DType> difgo(ws, Shape4(T, N, 4, H));
  Tensor<cpu, 2, DType> dh(ws + T * N * H * 4, Shape2(N, H));
  Tensor<cpu, 2, DType> dc(dh.dptr_ + N * H, Shape2(N, H));
  Tensor<cpu, 2, DType> htmp(dc.dptr_ + N * H, Shape2(N, H));
  const int offset = bid ? H : 0;
  const DType alpha = 1.0;
  const DType beta0 = 0.0;
  const DType beta1 = 1.0;
  const int cell_size = N * H;
  if (dhy_ptr != NULL) {
    memcpy(dh.dptr_, dhy_ptr, cell_size * sizeof(DType));
  }
  if (dcy_ptr != NULL) {
    memcpy(dc.dptr_, dcy_ptr, cell_size * sizeof(DType));
  }

  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  for (int i = T - 1; i >= 0; --i) {
    int t = bid ? T - 1 - i : i;
    int tnext = bid ? t + 1 : t - 1;
    const Tensor<cpu, 2, DType>& dhnext = i ? dh : dhx;
    const Tensor<cpu, 2, DType>& dcnext = i ? dc : dcx;
    const Tensor<cpu, 2, DType>& hnext = i ? htmp : hx;
    const Tensor<cpu, 2, DType>& cnext = i ? c[i - 1] : cx;
    #pragma omp parallel for num_threads(omp_threads)
    for (int jk = 0; jk < cell_size; ++jk) {
      int j = jk / H;
      int k = jk % H;
      DType tc = tanh(c[i][j][k]);
      DType it = ifgo[i][j][k][0];
      DType ft = ifgo[i][j][k][1];
      DType gt = ifgo[i][j][k][2];
      DType ot = ifgo[i][j][k][3];
      dh[j][k] += dy[t][j][k + offset];
      dc[j][k] += dh[j][k] * ot * (1 - tc * tc);
      difgo[t][j][0][k] = dc[j][k] * gt * it * (1 - it);
      difgo[t][j][1][k] = dc[j][k] * cnext[j][k] * ft * (1 - ft);
      difgo[t][j][2][k] = dc[j][k] * it * (1 - gt * gt);
      difgo[t][j][3][k] = dh[j][k] * tc * ot * (1 - ot);
      dcnext[j][k] = dc[j][k] * ft;
      if (i) {
        htmp[j][k] = y[tnext][j][k + offset];
      }
    }
    Tensor<cpu, 2, DType> dyh(difgo[t].dptr_, Shape2(N, H * 4));
    linalg_gemm(dyh, wh, dhnext, alpha, beta0, false, false);
    linalg_gemm(dyh, hnext, dwh, alpha, beta1, true, false);
  }
  Tensor<cpu, 2, DType> dyx(difgo.dptr_, Shape2(T * N, H * 4));
  linalg_gemm(dyx, wx, dx, alpha, bid ? beta1 : beta0, false, false);
  linalg_gemm(dyx, x, dwx, alpha, beta0, true, false);
  const int row = T * N;
  const int col = H * 4;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
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
                  DType* dw_ptr,
                  DType* db_ptr) {
  const int total_layers = D * L;
  Tensor<cpu, 3, DType> hx(hx_ptr, Shape3(total_layers, N, H));
  Tensor<cpu, 3, DType> cx(cx_ptr, Shape3(total_layers, N, H));
  Tensor<cpu, 3, DType> dhx(dhx_ptr, Shape3(total_layers, N, H));
  Tensor<cpu, 3, DType> dcx(dcx_ptr, Shape3(total_layers, N, H));
  const int b_size = 2 * H * 4;
  const int r_size = D * T * N * H * 6;
  const int y_offset = T * N * H * 5;
  const int w_size1 = (I + H) * H * 4;      // first layer
  const int w_size2 = (D * H + H) * H * 4;  // other layers
  const int cell_size = N * H;
  DType* dy_tmp_ptr = ws + T * cell_size * 4 + cell_size * 3;
  for (int i = L - 1; i >= 0; --i) {
    const int input_size = i ? H * D : I;
    const int w_size = i ? w_size2 : w_size1;
    int idx = i * D;
    DType* w_cur_ptr = i ? w_ptr + (w_size1 + (i - 1) * w_size2) * D : w_ptr;
    DType* dw_cur_ptr = i ? dw_ptr + (w_size1 + (i - 1) * w_size2) * D : dw_ptr;
    DType* db_cur_ptr = db_ptr + i * b_size * D;
    DType* rs_cur_ptr = rs + i * r_size;
    DType* dhy_cur_ptr = dhy_ptr ? dhy_ptr + i * cell_size * D : NULL;
    DType* dcy_cur_ptr = dcy_ptr ? dcy_ptr + i * cell_size * D : NULL;
    Tensor<cpu, 3, DType> y(rs_cur_ptr + y_offset, Shape3(T, N, H * D));
    Tensor<cpu, 3, DType> dy(dy_ptr, Shape3(T, N, H * D));
    Tensor<cpu, 2, DType> x(i ? y.dptr_ - r_size : x_ptr, Shape2(T * N, input_size));
    Tensor<cpu, 2, DType> dx(i ? dy_tmp_ptr : dx_ptr, Shape2(T * N, input_size));
    LstmBackwardSingleLayer<DType>(ws, rs_cur_ptr, false, T, N, input_size, H,
                                   x, hx[idx], cx[idx], y, dy, dx, dhx[idx], dcx[idx],
                                   dhy_cur_ptr, dcy_cur_ptr, w_cur_ptr, dw_cur_ptr, db_cur_ptr);
    if (D == 2) {
      w_cur_ptr += w_size;
      dw_cur_ptr += w_size;
      db_cur_ptr += b_size;
      ++idx;
      dhy_cur_ptr = dhy_ptr ? dhy_cur_ptr + cell_size : NULL;
      dcy_cur_ptr = dcy_ptr ? dcy_cur_ptr + cell_size : NULL;
      LstmBackwardSingleLayer<DType>(ws, rs_cur_ptr, true, T, N, input_size, H,
                                     x, hx[idx], cx[idx], y, dy, dx, dhx[idx], dcx[idx],
                                     dhy_cur_ptr, dcy_cur_ptr, w_cur_ptr, dw_cur_ptr, db_cur_ptr);
    }
    dy_ptr = dx.dptr_;
  }
}
#endif  // MXNET_OPERATOR_RNN_IMPL_H_
