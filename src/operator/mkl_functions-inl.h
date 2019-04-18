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
 * \file mkl_functions-inl.h
 * \brief
 * \author
*/
#ifndef MXNET_OPERATOR_MKL_FUNCTIONS_INL_H_
#define MXNET_OPERATOR_MKL_FUNCTIONS_INL_H_

#if MSHADOW_USE_MKL == 1
#include "mkl.h"

namespace mxnet {
namespace op {
namespace mkl_func {

MSHADOW_XINLINE
static bool check_size(const size_t n) {
  const size_t MKL_INT_MAX = (sizeof(MKL_INT) == sizeof(int)) ? INT_MAX : LLONG_MAX;
  return (n <= MKL_INT_MAX);
}

MSHADOW_XINLINE
static bool check_type(const int t) {
  return (t == mshadow::kFloat32 || t == mshadow::kFloat64);
}

#define MXNET_MKL_UNARY_MATH_FUNC(name, func)                                               \
struct name {                                                                               \
  MSHADOW_XINLINE static void Vectorize(const index_t n, const float *src, float *dst) {    \
    vs##func(static_cast<MKL_INT>(n), src, dst);                                            \
  }                                                                                         \
  MSHADOW_XINLINE static void Vectorize(const index_t n, const double *src, double *dst) {  \
    vd##func(static_cast<MKL_INT>(n), src, dst);                                            \
  }                                                                                         \
};

#define MXNET_MKL_BINARY_MATH_FUNC(name, func)                                        \
struct name {                                                                         \
  MSHADOW_XINLINE static void Vectorize(const index_t n,                              \
                                        const float *a,                               \
                                        const float *b,                               \
                                        float *c) {                                   \
    vs##func(static_cast<MKL_INT>(n), a, b, c);                                       \
  }                                                                                   \
  MSHADOW_XINLINE static void Vectorize(const index_t n,                              \
                                        const double *a,                              \
                                        const double *b,                              \
                                        double *c) {                                  \
    vd##func(static_cast<MKL_INT>(n), a, b, c);                                       \
  }                                                                                   \
};

MXNET_MKL_UNARY_MATH_FUNC(erf, Erf);
MXNET_MKL_UNARY_MATH_FUNC(exp, Exp);
MXNET_MKL_UNARY_MATH_FUNC(exp2, Exp2);
MXNET_MKL_UNARY_MATH_FUNC(exp10, Exp10);
MXNET_MKL_UNARY_MATH_FUNC(expm1, Expm1);
MXNET_MKL_UNARY_MATH_FUNC(log, Ln);
MXNET_MKL_UNARY_MATH_FUNC(log2, Log2);
MXNET_MKL_UNARY_MATH_FUNC(log10, Log10);
MXNET_MKL_UNARY_MATH_FUNC(log1p, Log1p);

MXNET_MKL_UNARY_MATH_FUNC(sin, Sin);
MXNET_MKL_UNARY_MATH_FUNC(cos, Cos);
MXNET_MKL_UNARY_MATH_FUNC(tan, Tan);
MXNET_MKL_UNARY_MATH_FUNC(asin, Asin);
MXNET_MKL_UNARY_MATH_FUNC(acos, Acos);
MXNET_MKL_UNARY_MATH_FUNC(atan, Atan);

MXNET_MKL_UNARY_MATH_FUNC(sinh, Sinh);
MXNET_MKL_UNARY_MATH_FUNC(cosh, Cosh);
MXNET_MKL_UNARY_MATH_FUNC(tanh, Tanh);
MXNET_MKL_UNARY_MATH_FUNC(asinh, Asinh);
MXNET_MKL_UNARY_MATH_FUNC(acosh, Acosh);
MXNET_MKL_UNARY_MATH_FUNC(atanh, Atanh);

MXNET_MKL_UNARY_MATH_FUNC(sqrt, Sqrt);
MXNET_MKL_UNARY_MATH_FUNC(abs, Abs);
MXNET_MKL_UNARY_MATH_FUNC(cbrt, Cbrt);
MXNET_MKL_UNARY_MATH_FUNC(round, Round);
MXNET_MKL_UNARY_MATH_FUNC(ceil, Ceil);
MXNET_MKL_UNARY_MATH_FUNC(floor, Floor);
MXNET_MKL_UNARY_MATH_FUNC(trunc, Trunc);

MXNET_MKL_UNARY_MATH_FUNC(lgamma, LGamma);
MXNET_MKL_UNARY_MATH_FUNC(tgamma, TGamma);
MXNET_MKL_UNARY_MATH_FUNC(square, Sqr);

MXNET_MKL_BINARY_MATH_FUNC(add, Add);
MXNET_MKL_BINARY_MATH_FUNC(sub, Sub);
MXNET_MKL_BINARY_MATH_FUNC(mul, Mul);
MXNET_MKL_BINARY_MATH_FUNC(pow, Pow);
MXNET_MKL_BINARY_MATH_FUNC(hypot, Hypot);


template <typename DType>
MSHADOW_XINLINE static void sub_(index_t n, DType *in, DType b, DType *dst) {
  for (index_t i = 0; i < n; i++)
    dst[i] = in[i] - b;
}

template <typename DType>
MSHADOW_XINLINE static void div_(index_t n, DType *in, DType b, DType *dst) {
  for (index_t i = 0; i < n; i++)
    dst[i] = in[i] / b;
}

template <typename DType>
MSHADOW_XINLINE static void sum_(index_t n, DType *in, DType *dst) {
  // dst[0] = cblas_sasum(n, in, 1);
  DType sum = 0.0f;
  for (index_t i = 0; i < n; i++)
    sum += in[i];

  dst[0] = sum;
}

template <typename DType>
MSHADOW_XINLINE static void max_(int n, DType * __restrict__ in, DType *dst) {
  dst[0] = in[0];
  for (int i = 1; i < n; i++)
    dst[0] = (dst[0] < in[i]) ? in[i] : dst[0];
}

// LayerNorm on the last dimension
template <typename DType>
MSHADOW_XINLINE static void LayerNormLastDim(const index_t m,
                                             const index_t n,
                                             const DType *a,
                                             const DType *b,
                                             const DType *ws,
                                             const DType *gamma,
                                             const DType *beta,
                                             const DType *mean,
                                             const DType *var,
                                             const DType eps) {
#pragma omp parallel for
  for (index_t i = 0; i < m; i++) {
    DType* in_offset = a + i * n;
    DType* out_offset = b + i * n;
    DType* ws_offset = ws + i * n;

    sum_(n, in_offset, &(mean[i]));
    mean[i] /= n;
    sub_(n, in_offset, mean[i], out_offset);
    square(n, out_offset, ws_offset);
    sum_(n, ws_offset, &(var[i]));
    var[i] = sqrt(var[i] / n + eps);

    mul(n, out_offset, gamma, out_offset);
    div_(n, out_offset, var[i], out_offset);
    add(n, out_offset, beta, out_offset);
  }
}

// softmax on the last dimension
template <typename DType>
MSHADOW_XINLINE static void SoftmaxLastDim(const index_t m,
                                           const index_t n,
                                           const DType *a,
                                           const DType *b) {
#pragma omp paralle for
  for (index_t i = 0; i < m; i++) {
    DType* in_offset = a + i * n;
    DType* out_offset = b + i * n;

    exp(n, in_offset, out_offset);
    float sum = 0.0f;
    sum_(n, out_offset, &sum);
    div_(n, out_offset, sum, out_offset);
  }
}

template <typename DType>
MSHADOW_XINLINE static void LogSoftmaxLastDim(const index_t m,
                                              const index_t n,
                                              const DType *a,
                                              const DType *b) {
#pragma parallel for
  for (index_t i = 0; i < m; i++) {
    DType* in_offset = a + i * n;
    DType* out_offset = b + i * n;

    DType b, logsum;
    max_(n, in_offset, &b);
    sub_(n, in_offset, b, out_offset);
    exp(n, out_offset, out_offset);
    sum_(n, out_offset, &logsum);
    logsum = b + logf(logsum);
    sub_(n, in_offset, logsum, out_offset);
  }
}

}  // namespace mkl_func
}  // namespace op
}  // namespace mxnet
#endif  // MSHADOW_USE_MKL == 1
#endif  // MXNET_OPERATOR_MKL_FUNCTIONS_INL_H_
