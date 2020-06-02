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
__device__ inline DTypeGrad backward_relu(const DTypeGrad grad, const DType val) {
  if (isnan(val)) return val;
  return val > 0 ? grad : 0;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sigmoid(const DTypeGrad grad, const DType out) {
  return grad * out * (1 - out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_softrelu(const DTypeGrad grad, const DType val) {
  return grad * sigmoid(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_softsign(const DTypeGrad grad, const DType val) {
  const DType ap1 = 1 + op::abs(val);
  return grad / (ap1 * ap1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_abs(const DTypeGrad grad, const DType val) {
  return grad * op::sign(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_exp(const DTypeGrad grad, const DType val) {
  return grad * op::exp(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_expm1(const DTypeGrad grad, const DType val) {
  return backward_exp(grad, val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log(const DTypeGrad grad, const DType val) {
  return grad / val;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log10(const DTypeGrad grad, const DType val) {
  return grad / (val * op::log(static_cast<DTypeGrad>(10)));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log2(const DTypeGrad grad, const DType val) {
  return grad / (val * op::log(static_cast<DTypeGrad>(2)));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_log1p(const DTypeGrad grad, const DType val) {
  return grad / (1 + val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sin(const DTypeGrad grad, const DType val) {
  return grad * op::cos(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cos(const DTypeGrad grad, const DType val) {
  return -grad * op::sin(val);
}

// Uses output from tan
template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_tan(const DTypeGrad grad, const DType out) {
  return grad * (out * out + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arcsin(const DTypeGrad grad, const DType val) {
  return grad / op::sqrt(1 - val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arccos(const DTypeGrad grad, const DType val) {
  return -grad / op::sqrt(1 - val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arctan(const DTypeGrad grad, const DType val) {
  return grad / (1 + val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_degrees(const DTypeGrad grad, const DType /* val */) {
  return op::degrees(grad);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_radians(const DTypeGrad grad, const DType /* val */) {
  return op::radians(grad);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sinh(const DTypeGrad grad, const DType val) {
  return grad * op::cosh(val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cosh(const DTypeGrad grad, const DType val) {
  return grad * op::sinh(val);
}

// Uses tanh output
template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_tanh(const DTypeGrad grad, const DType out) {
  return grad * (1 - out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arcsinh(const DTypeGrad grad, const DType val) {
  return grad / op::sqrt(val * val + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arccosh(const DTypeGrad grad, const DType val) {
  return grad / op::sqrt(val * val - 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_arctanh(const DTypeGrad grad, const DType val) {
  return grad / (1 - val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_sqrt(const DTypeGrad grad, const DType out) {
  return 0.5 * grad / out;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_rsqrt(const DTypeGrad grad, const DType val) {
  const DType inv = 1 / val;
  return -0.5 * grad * op::sqrt(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_cbrt(const DTypeGrad grad, const DType out) {
  return grad / (3.0f * out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_rcbrt(const DTypeGrad grad, const DType val) {
  const DType inv = 1 / val;
  return -1.f/3.f * grad * op::cbrt(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_square(const DTypeGrad grad, const DType val) {
  return 2 * val * grad;
}

template <typename DType, typename DType2>
__device__ inline DType rdiv_grad(const DType val,
                                  const DType2 val2) {
  return -val2 / (val * val);
}

template <typename DType, typename DType2>
__device__ inline DType mod_grad(const DType val,
                                 const DType2 val2) {
  if (type_util::is_integral<DType>::value) {
    return 0;
  } else {
    return 1;
  }
}

template <typename DType, typename DType2>
__device__ inline DType rmod_grad(const DType val,
                                  const DType2 val2) {
  if (type_util::is_integral<DType>::value) {
    return 0;
  } else {
    return -op::floor(val2 / val);
  }
}

template <typename DType, typename DType2>
__device__ inline DType power_grad(const DType val,
                                   const DType2 val2) {
  return op::power(val, val2 - 1.f) * val2;
}

template <typename DType, typename DType2>
__device__ inline DType rpower_grad(const DType val,
                                   const DType2 val2) {
  return val * op::log(val2);
}

template <typename DType, typename DType2>
__device__ inline DType hypot_grad_left(const DType val,
                                        const DType2 val2) {
  return val / op::hypot(val, val2);
}

template <typename DType, typename DType2>
__device__ inline DType hypot_grad_right(const DType val,
                                         const DType2 val2) {
  return val2 / op::hypot(val, val2);
}

template <typename DType, typename DType2>
__device__ inline DType copysign_grad(const DType val,
                                      const DType2 val2) {
  return (a >= 0 && b >= 0) || (a < 0 && b < 0) ? 1 : -1;
}

template <typename DType, typename DType2>
__device__ inline DType arctan2_grad(const DType val,
                                     const DType2 val2) {
  return val2 / (val * val + val2 * val2);
}

template <typename DType, typename DType2>
__device__ inline DType rarctan2_grad(const DType val,
                                      const DType2 val2) {
  return val / (val * val + val2 * val2);
}

template <typename DType, typename DType2>
__device__ inline DType ldexp_grad(const DType val,
                                   const DType2 val2) {
  return op::power(static_cast<DType>(2), val2);
}

template <typename DType, typename DType2>
__device__ inline DType rldexp_grad(const DType val,
                                    const DType2 val2) {
  returni val2 * op::power(static_cast<DType>(2), val) * op::log(static_cast<DType>(2));
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_clip(const DTypeGrad grad, const DType val,
                                          const float a_min, const float a_max) {
  if (val > a_max || val < a_min) {
    return 0;
  } else {
    return grad;
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_reciprocal(const DTypeGrad grad, const DType val) {
  return -grad / (val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_erf(const DTypeGrad grad, const DType val) {
  return 2.0f / op::sqrt(pi) * op::exp(-(val*val)) * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_erfinv(const DTypeGrad grad, const DType val) {
  return 0.5f * op::sqrt(pi) * op::exp(val * val) * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_gamma(const DTypeGrad grad, const DType val) {
  if (type_util::is_same<DTypeGrad, double>::value) {
    return grad * op::gamma(val) * op::special_functions::cephes::psi<double>(val);
  } else {
    return grad * op::gamma(val) * op::special_functions::cephes::psi<float>(val);
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline DTypeGrad backward_gammaln(const DTypeGrad grad, const DType val) {
  if (type_util::is_same<DTypeGrad, double>::value) {
    return grad * op::special_functions::cephes::psi<double>(val);
  } else {
    return grad * op::special_functions::cephes::psi<float>(val);
  }
}

template <typename DType, typename DType2>
__device__ inline DType smooth_l1_grad(const DType val, const DType2 scalar) {
  auto bsq = scalar * scalar;
  auto ibsq = 1.0f / bsq;
  if (val > ibsq) {
    return 1;
  } else if (val < -ibsq) {
    return -1;
  } else {
    return bsq * val;
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
