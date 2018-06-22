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
 * Copyright (c) 2018 by Contributors
 * \file cublas_transpose.h
 * \brief Transpose cuBLAS Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko
 */
#ifndef MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_CUBLAS_TRANSPOSE_H_
#define MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_CUBLAS_TRANSPOSE_H_

#include "../../../common/cuda_utils.h"

// Transpose using cuBLAS approach.
// @param1 cublas_handle: cuBLAS Handle
// @param2 _o: Output
// [cols x rows]
// @param2 _1: 1st Operand
// [rows x cols]
// @param2 rows: Number of Rows
// @param3 cols: Number of Cols
inline void transpose(
  const cublasHandle_t cublas_handle,
        float * const _o,
  const float * const _1,
  const unsigned & rows, const unsigned & cols
  ) {
  static const float alpha = 1.0, beta = 0.0;

  // cuBLAS Matrix Transpose - C = alpha * op(A) + beta * op(B)
  // @param1 handle: cuBLAS Handle
  // @param2 transa: Transpose matrix A?
  // @param3 transb: Transpose matrix B?
  // @param4 alpha
  // @param5  A
  // @param6  lda: Leading Dimension of A
  // @param7 beta
  // @param8  B
  // @param9  ldb: Leading Dimension of B
  // @param10 C
  // @param11 ldc: Leading Dimension of C
  CUBLAS_CALL(cublasSgeam(cublas_handle,
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          rows, cols,
                          &alpha, _1, cols,
                          &beta, nullptr, rows,
                          _o, rows));
}

#endif  // MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_CUBLAS_TRANSPOSE_H_

