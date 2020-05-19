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

#ifndef MXNET_COMMON_CUDA_RTC_BACKWARD_FUNCTIONS_INL_H_
#define MXNET_COMMON_CUDA_RTC_BACKWARD_FUNCTIONS_INL_H_

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char backward_function_definitions[] = R"code(

namespace op {

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_relu(const DType val, const DTypeGrad grad) {
  return val > 0 ? grad : 0;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sigmoid(const DType out, const DTypeGrad grad) {
  return grad * out * (1 - out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_softrelu(const DType val, const DTypeGrad grad) {
  return grad * sigmoid(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_softsign(const DType val, const DTypeGrad grad) {
  const DType ap1 = 1 + fabsf(val);
  return grad / (ap1 * ap1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_exp(const DType val, const DTypeGrad grad) {
  return grad * expf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_expm1(const DType val, const DTypeGrad grad) {
  return grad * expf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log(const DType val, const DTypeGrad grad) {
  return grad / val;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log10(const DType val, const DTypeGrad grad) {
  return grad / (val * logf(10));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log2(const DType val, const DTypeGrad grad) {
  return grad / (val * logf(2));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log1p(const DType val, const DTypeGrad grad) {
  return grad / (1 + val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sin(const DType val, const DTypeGrad grad) {
  return grad * cosf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cos(const DType val, const DTypeGrad grad) {
  return -grad * sinf(val);
}

// Uses output from tan
template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_tan(const DType out, const DTypeGrad grad) {
  return grad * (out * out + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arcsin(const DType val, const DTypeGrad grad) {
  return grad / sqrtf(1 - val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arccos(const DType val, const DTypeGrad grad) {
  return -grad / sqrtf(1 - val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arctan(const DType val, const DTypeGrad grad) {
  return grad / (1 + val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sinh(const DType val, const DTypeGrad grad) {
  return grad * coshf(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cosh(const DType val, const DTypeGrad grad) {
  return grad * sinhf(val);
}

// Uses tanh output
template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_tanh(const DType out, const DTypeGrad grad) {
  return grad * (1 - out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arcsinh(const DType val, const DTypeGrad grad) {
  return grad / sqrtf(val * val + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arccosh(const DType val, const DTypeGrad grad) {
  return grad / sqrtf(val * val - 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arctanh(const DType val, const DTypeGrad grad) {
  return grad / (1 - val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sqrt(const DType out, const DTypeGrad grad) {
  return 0.5 * grad / out;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_rsqrt(const DType val, const DTypeGrad grad) {
  const DType inv = 1 / val;
  return -0.5 * grad * sqrtf(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cbrt(const DType out, const DTypeGrad grad) {
  return grad / (3.0f * out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_rcbrt(const DType val, const DTypeGrad grad) {
  const DType inv = 1 / val;
  return -1.f/3.f * grad * cbrtf(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_square(const DType val, const DTypeGrad grad) {
  return 2 * val * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_clip(const DType val, const DTypeGrad grad,
                                          const float a_min, const float a_max) {
  if (val > a_max || val < a_min) {
    return 0;
  } else {
    return grad;
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_reciprocal(const DType val, const DTypeGrad grad) {
  return -grad / (val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_erf(const DType val, const DTypeGrad grad) {
  return 2.0f / sqrt(pi) * exp(-(val*val)) * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_erfinv(const DType val, const DTypeGrad grad) {
  return 0.5f * sqrt(pi) * exp(val * val) * grad;
}

template <typename DType, typename DType2, typename DTypeGrad>
__device__ inline DTypeGrad backward_smooth_l1(const DType val, const DType2 scalar,
                                               const DTypeGrad grad) {
  auto bsq = scalar * scalar;
  auto ibsq = 1.0f / bsq;
  if (val > ibsq) {
    return grad;
  } else if (val < -ibsq) {
    return -grad;
  } else {
    return bsq * val * grad;
  }
}

}  // namespace op

)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_BACKWARD_FUNCTIONS_INL_H_
