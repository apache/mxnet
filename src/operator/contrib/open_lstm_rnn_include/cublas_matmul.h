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
 * \file cublas_transpose.cuh
 * \brief Matmul cuBLAS Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko
 */
#ifndef MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_CUBLAS_MATMUL_H_
#define MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_CUBLAS_MATMUL_H_

#include "../../../common/cuda_utils.h"

// Matmul using cuBLAS approach.
// @param1 cublas_handle: cuBLAS Handle
// @param2 _o: Output
// [_1_rows x _2_cols]
// @param3 _1: 1st Operand
// [_1_rows x _1_cols_2_rows]
// @param4 _2: 2nd Operand
// [_1_cols_2_rows x _2_cols]
// @param5 _1_rows: Rows of 1st Operand
// @param6 _2_cols: Cols of 2nd Operand
// @param7 _1_cols_2_rows: Cols of 1st Operand (or Rows of 2nd Operand)
inline void matmul
  (
    const cublasHandle_t cublas_handle,
          float * const _o,
    const float * const _1,
    const float * const _2,
    const unsigned & _1_rows, const unsigned & _2_cols,
    const unsigned & _1_cols_2_rows
  ) {
  static const float alpha = 1.0, beta = 0.0;

  // cuBLAS Matrix Multiply - C = alpha * op(A) * op(B) + beta * C
  // @param1  handle: cuBLAS Handle
  // @param2  transa: Transpose matrix A?
  // @param3  transb: Transpose matrix B?
  // @param4  m: Number of Rows of Matrix A and C
  // @param5  n: Number of Cols of Matrix B and C
  // @param6  k: Number of Cols of Matrix A and Rows of B
  // @param7  alpha
  // @param8  A
  // @param9  lda: Leading Dimension of A
  // @param10 B
  // @param11 ldb: Leading Dimension of B
  // @param12 beta
  // @param13 C
  // @param14 ldc: Leading Dimension of C
  CUBLAS_CALL(cublasSgemm(cublas_handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    _2_cols, _1_rows, _1_cols_2_rows,
    &alpha,
    _2, _2_cols,
    _1, _1_cols_2_rows,
    &beta,
    _o, _2_cols));
}
// Matmul using cuBLAS approach (Extended Version).
// @param1  cublas_handle: cuBLAS Handle
// @param2  _o: Output
// [_1_rows x _2_cols]
// @param3  _1: 1st Operand
// [_1_rows x _1_cols_2_rows]
// @param4  _2: 2nd Operand
// [_1_cols_2_rows x _2_cols]
// @param5  _1_op: Operation on 1st Operand (Normal? Transpose?)
// @param6  _2_op: Operation on 2nd Operand (Normal? Transpose?)
// @param7  _1_rows: Rows of 1st Operand
// @param8  _1_cols: Cols of 1st Operand
// @param9  _2_rows: Rows of 2nd Operand
// @param10 _2_cols: Cols of 2nd Operand
// @param11 alpha: see the comment below
// @param12  beta: see the comment below
inline void matmul_ex
  (
    const cublasHandle_t cublas_handle,
          float * const _o,
    const float * const _1,
    const float * const _2,
    const cublasOperation_t & _1_op,
    const cublasOperation_t & _2_op,
    const unsigned & _1_rows, const unsigned & _1_cols,
    const unsigned & _2_rows, const unsigned & _2_cols,
    const float & alpha, const float & beta
  ) {
  if (_1_op == CUBLAS_OP_N) {
    if (_2_op == CUBLAS_OP_N) {
      assert(_1_cols == _2_rows);
    } else {
      assert(_1_cols == _2_cols);
    }
  } else {
    if (_2_op == CUBLAS_OP_N) {
      assert(_1_rows == _2_rows);
    } else {
      assert(_1_rows == _2_cols);
    }
  }

  // cuBLAS Matrix Multiply - C = alpha * op(A) * op(B) + beta * C
  // @param1  handle: cuBLAS Handle
  // @param2  transa: Transpose matrix A?
  // @param3  transb: Transpose matrix B?
  // @param4  m: Number of Rows of Matrix op(A) and C
  // @param5  n: Number of Cols of Matrix op(B) and C
  // @param6  k: Number of Cols of Matrix op(A) and Rows of op(B)
  // @param7  alpha
  // @param8  A
  // @param9  lda: Leading Dimension of A
  // @param10 B
  // @param11 ldb: Leading Dimension of B
  // @param12 beta
  // @param13 C
  // @param14 ldc: Leading Dimension of C
  CUBLAS_CALL(cublasSgemm(cublas_handle,
    _2_op, _1_op,
    _2_op == CUBLAS_OP_N ? _2_cols : _2_rows,
    _1_op == CUBLAS_OP_N ? _1_rows : _1_cols,
    _1_op == CUBLAS_OP_N ? _1_cols : _1_rows,
    &alpha,
    _2, _2_cols,
    _1, _1_cols,
    &beta,
    _o, _2_op == CUBLAS_OP_N ? _2_cols : _2_rows));
}
// Strided Batched Matmul using cuBLAS approach.
// @param1  cublas_handle: cuBLAS Handle
// @param2  _o: Output
// [_1_rows x _2_cols]
// @param3  _1: 1st Operand
// [_1_rows x _1_cols_2_rows]
// @param4  _2: 2nd Operand
// [_1_cols_2_rows x _2_cols]
// @param5  _1_rows: Rows of 1st Operand
// @param6  _2_cols: Cols of 2nd Operand
// @param7  _1_cols_2_rows: Cols of 1st Operand (or Rows of 2nd Operand)
// @param8  stride_o: Output Stride
// @param9  stride_1: Stride of 1st Operand
// @param10 stride_2: Stride of 2nd Operand
// @param11 batch_cnt: Batch Count
inline void matmul_stridedbatched
  (
    const cublasHandle_t cublas_handle,
          float * const _o,
    const float * const _1,
    const float * const _2,
    const unsigned & _1_rows, const unsigned & _2_cols,
    const unsigned & _1_cols_2_rows,
    const unsigned & stride_o,
    const unsigned & stride_1,
    const unsigned & stride_2,
    const unsigned & batch_cnt
  ) {
  static const float alpha = 1.0, beta = 0.0;

  // Function argument list is very similar to the API above,
  // except that the operation is now strided batched.
  CUBLAS_CALL(cublasSgemmStridedBatched(cublas_handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    _2_cols, _1_rows, _1_cols_2_rows,
    &alpha,
    _2, _2_cols, stride_2,
    _1, _1_cols_2_rows, stride_1,
    &beta,
    _o, _2_cols, stride_o,
    batch_cnt));
}
// Strided Batched Matmul using cuBLAS approach (Extended Version).
// @param1  cublas_handle: cuBLAS Handle
// @param2  _o: Output
// [_1_rows x _2_cols]
// @param3  _1: 1st Operand
// [_1_rows x _1_cols_2_rows]
// @param4  _2: 2nd Operand
// [_1_cols_2_rows x _2_cols]
// @param5  _1_op: Operation on 1st Operand (Normal? Transpose?)
// @param6  _1_op: Operation on 1st Operand (Normal? Transpose?)
// @param7  _1_rows: Rows of 1st Operand
// @param8  _1_cols: Cols of 1st Operand
// @param9  _2_rows: Rows of 2nd Operand
// @param10  _2_cols: Cols of 2nd Operand
// @param11 stride_o: Output Stride
// @param12 stride_1: Stride of 1st Operand
// @param13 stride_2: Stride of 2nd Operand
// @param14 batch_cnt: Batch Count
// @param15 alpha: refer to matmul_ex
// @param16  beta: refer to matmul_ex
inline void matmul_stridedbatched_ex(
    const cublasHandle_t cublas_handle,
          float * const _o,
    const float * const _1,
    const float * const _2,
    const cublasOperation_t & _1_op,
    const cublasOperation_t & _2_op,
    const unsigned & _1_rows, const unsigned & _1_cols,
    const unsigned & _2_rows, const unsigned & _2_cols,
    const unsigned & stride_o,
    const unsigned & stride_1,
    const unsigned & stride_2,
    const unsigned & batch_cnt,
    const float & alpha, const float & beta
  ) {
  if (_1_op == CUBLAS_OP_N) {
    if (_2_op == CUBLAS_OP_N) {
      assert(_1_cols == _2_rows);
    } else {
      assert(_1_cols == _2_cols);
    }
  } else {
    if (_2_op == CUBLAS_OP_N) {
      assert(_1_rows == _2_rows);
    } else {
      assert(_1_rows == _2_cols);
    }
  }

  // Function argument list is very similar to the API above,
  // except that the operation is now strided batched.
  CUBLAS_CALL(cublasSgemmStridedBatched(cublas_handle,
    _2_op, _1_op,
    _2_op == CUBLAS_OP_N ? _2_cols : _2_rows,
    _1_op == CUBLAS_OP_N ? _1_rows : _1_cols,
    _1_op == CUBLAS_OP_N ? _1_cols : _1_rows,
    &alpha,
    _2, _2_cols, stride_2,
    _1, _1_cols, stride_1,
    &beta,
    _o, _2_op == CUBLAS_OP_N ? _2_cols : _2_rows, stride_o,
    batch_cnt));
}

#endif  // MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_CUBLAS_MATMUL_H_
