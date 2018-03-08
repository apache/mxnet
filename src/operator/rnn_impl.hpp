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
inline DType sigmoid(DType x){
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
    Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, 4 * H));
    Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, 4 * H));

    Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
    Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
    Tensor<cpu, 3, DType> h(rs, Shape3(T, N, H));
    Tensor<cpu, 3, DType> c(rs + T * N * H, Shape3(T, N, H)); 
    Tensor<cpu, 4, DType> ifgo(rs + T * N * H * 2, Shape4(T, N, H, 4));
    DType alpha = 1.0;
    DType beta = 0.0;
    linalg_gemm(x, wx, yx_flat, alpha, beta, false, true); 
    
    for (int i = 0; i < T; ++i) {
        linalg_gemm((i == 0) ? hx : h[i-1], wh, yh_flat, alpha, beta, false, true);
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < H; ++k) {
                 DType it = sigmoid<DType>(yx[i][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
                 DType ft =           tanh(yx[i][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
                 DType gt = sigmoid<DType>(yx[i][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
                 DType ot = sigmoid<DType>(yx[i][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
                 DType ct = ((i == 0) ? cx[j][k] : c[i-1][j][k]) * ft + it * gt; 
                 h[i][j][k] = ot * tanh(ct);
                 c[i][j][k] = ct;
                 //reserve
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
    Tensor<cpu, 2, DType> yx_flat(ws, Shape2(T * N, 4 * H));
    Tensor<cpu, 2, DType> yh_flat(ws + T * N * H * 4, Shape2(N, 4 * H));
    Tensor<cpu, 2, DType> c(yh_flat.dptr_ + N * H * 4, Shape2(N, H)); 

    Tensor<cpu, 4, DType> yx(yx_flat.dptr_, Shape4(T, N, 4, H));
    Tensor<cpu, 3, DType> yh(yh_flat.dptr_, Shape3(N, 4, H));
    Tensor<cpu, 3, DType> h(y_ptr, Shape3(T, N, H));
    DType alpha = 1.0;
    DType beta = 0.0;
    linalg_gemm(x, wx, yx_flat, alpha, beta, false, true); 
    
    for (int i = 0; i < T; ++i) {
        linalg_gemm((i == 0) ? hx : h[i-1], wh, yh_flat, alpha, beta, false, true);
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < H; ++k) {
                 DType it = sigmoid<DType>(yx[i][j][0][k] + yh[j][0][k] + bx[0][k] + bh[0][k]);
                 DType ft =           tanh(yx[i][j][1][k] + yh[j][1][k] + bx[1][k] + bh[1][k]);
                 DType gt = sigmoid<DType>(yx[i][j][2][k] + yh[j][2][k] + bx[2][k] + bh[2][k]);
                 DType ot = sigmoid<DType>(yx[i][j][3][k] + yh[j][3][k] + bx[3][k] + bh[3][k]);
                 DType ct = ((i == 0) ? cx[j][k] : c[j][k]) * ft + it * gt; 
                 h[i][j][k] = ot * tanh(ct);
                 c[j][k] = ct;
            }
        }
    }
    if (state_outputs) {
        memcpy(hy_ptr, y_ptr + (T - 1) * N * H, N * H * sizeof(float));
        memcpy(cy_ptr, c.dptr_, N * H * sizeof(float)); 
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
void RNNForwardTraining(DType* ws,
                        DType* rs,
                        bool state_outputs,
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
                        DType* y_ptr,
                        DType* hy_ptr,
                        DType* cy_ptr) {
    LstmForwardTraining<DType>(ws, 
                               rs, 
                               state_outputs,
                               num_layers,
                               direction,
                               seq_length,
                               batch_size,
                               input_size,
                               state_size,
                               x_ptr,
                               hx_ptr,
                               cx_ptr,
                               w_ptr,
                               y_ptr,
                               hy_ptr,
                               cy_ptr);
}
template <typename DType>
void RNNForwardInference(DType* ws,
                         bool state_outputs,
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
                         DType* y_ptr,
                         DType* hy_ptr,
                         DType* cy_ptr) {
    LstmForwardInference<DType>(ws, 
                                state_outputs,
                                num_layers,
                                direction,
                                seq_length,
                                batch_size,
                                input_size,
                                state_size,
                                x_ptr,
                                hx_ptr,
                                cx_ptr,
                                w_ptr,
                                y_ptr,
                                hy_ptr,
                                cy_ptr);
}

template <typename DType>
void RNNBackward(DType* ws,
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
                 DType* dy_ptr,
                 DType* dhy_ptr,
                 DType* dcy_ptr,
                 DType* dx_ptr,
                 DType* dhx_ptr,
                 DType* dcx_ptr,
                 DType* dw_ptr) {
}
