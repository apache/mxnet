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
__device__ inline mixed_type<DTypeGrad, DType>
backward_relu(const DTypeGrad grad, const DType val) {
  if (isnan(val)) return val;
  return val > 0 ? grad : 0;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_sigmoid(const DTypeGrad grad, const DType out) {
  return grad * out * (1 - out);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_softrelu(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad * sigmoid(v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_softsign(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  const auto ap1 = 1 + op::abs(v);
  return grad / (ap1 * ap1);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_abs(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad * op::sign(v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_exp(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad * op::exp(v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_expm1(const DTypeGrad grad, const DType val) {
  return backward_exp(grad, val);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_log(const DTypeGrad grad, const DType val) {
  return grad / val;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_log10(const DTypeGrad grad, const DType val) {
  return grad / (val * op::log(static_cast<DTypeGrad>(10)));
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_log2(const DTypeGrad grad, const DType val) {
  return grad / (val * op::log(static_cast<DTypeGrad>(2)));
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_log1p(const DTypeGrad grad, const DType val) {
  return grad / (1 + val);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_sin(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad * op::cos(v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_cos(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return -grad * op::sin(v);
}

// Uses output from tan
template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_tan(const DTypeGrad grad, const DType out) {
  return grad * (out * out + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_arcsin(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad / op::sqrt(1 - v*v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_arccos(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return -grad / op::sqrt(1 - v*v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_arctan(const DTypeGrad grad, const DType val) {
  return grad / (1 + val*val);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_degrees(const DTypeGrad grad, const DType /* val */) {
  return op::degrees(grad);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_radians(const DTypeGrad grad, const DType /* val */) {
  return op::radians(grad);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_sinh(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad * op::cosh(v);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_cosh(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad * op::sinh(v);
}

// Uses tanh output
template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_tanh(const DTypeGrad grad, const DType out) {
  return grad * (1 - out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_arcsinh(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad / op::sqrt(v * v + 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_arccosh(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  return grad / op::sqrt(v * v - 1);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_arctanh(const DTypeGrad grad, const DType val) {
  return grad / (1 - val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_sqrt(const DTypeGrad grad, const DType out) {
  return 0.5 * grad / out;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_rsqrt(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  const auto inv = 1 / v;
  return -0.5 * grad * op::sqrt(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_cbrt(const DTypeGrad grad, const DType out) {
  return grad / (3.0f * out * out);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_rcbrt(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  const auto inv = 1 / v;
  return -1.f/3.f * grad * op::cbrt(inv) * inv;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_square(const DTypeGrad grad, const DType val) {
  return 2 * val * grad;
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
rdiv_grad(const DType val,
          const DType2 val2) {
  return -val2 / (val * val);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
div_grad(const DType val,
         const DType2 val2) {
  const mixed_type<DType, DType2> temp = val2;
  return op::reciprocal(temp);
}

template <typename DType, typename DType2>
__device__ inline DType div_rgrad(const DType val,
                                  const DType2 val2) {
  return -val / (val2 * val2);
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
__device__ inline DType mod_rgrad(const DType val,
                                  const DType2 val2) {
  if (type_util::is_integral<DType>::value) {
    return 0;
  } else {
    return -op::floor(val / val2);
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
__device__ inline mixed_type<DType, DType2>
power_grad(const DType val,
           const DType2 val2) {
  return op::power(val, val2 - 1.f) * val2;
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
power_rgrad(const DType val,
            const DType2 val2) {
  const mixed_type<DType, DType2> temp = val;
  return op::power(val, val2) * op::log(temp);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
rpower_grad(const DType val,
            const DType2 val2) {
  const mixed_type<DType, DType2> temp = val2;
  return val * op::log(temp);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
hypot_grad_left(const DType val,
                const DType2 val2) {
  return val / op::hypot(val, val2);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
hypot_grad_right(const DType val,
                 const DType2 val2) {
  return val2 / op::hypot(val, val2);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
copysign_grad(const DType val,
              const DType2 val2) {
  return (val >= 0 && val2 >= 0) || (val < 0 && val2 < 0) ? 1 : -1;
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
arctan2_grad(const DType val,
             const DType2 val2) {
  return val2 / (val * val + val2 * val2);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
rarctan2_grad(const DType val,
              const DType2 val2) {
  return val / (val * val + val2 * val2);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
arctan2_rgrad(const DType val,
              const DType2 val2) {
  return -rarctan2_grad(val, val2);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
ldexp_grad(const DType val,
           const DType2 val2) {
  return op::power(static_cast<DType>(2), val2);
}

template <typename DType, typename DType2>
__device__ inline mixed_type<DType, DType2>
rldexp_grad(const DType val,
            const DType2 val2) {
  using type = mixed_type<DType, DType2>;
  return val2 * op::power(static_cast<type>(2), val) * op::log(static_cast<type>(2));
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_clip(const DTypeGrad grad, const DType val,
              const float a_min, const float a_max) {
  if (val > a_max || val < a_min) {
    return 0;
  } else {
    return grad;
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_reciprocal(const DTypeGrad grad, const DType val) {
  return -grad / (val * val);
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_erf(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  constexpr mixed_type<DTypeGrad, DType> my_pi = pi;
  return 2.0f / op::sqrt(my_pi) * op::exp(-(v*v)) * grad;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_erfinv(const DTypeGrad grad, const DType val) {
  constexpr mixed_type<DTypeGrad, DType> my_pi = pi;
  const mixed_type<DTypeGrad, DType> g = grad;
  const mixed_type<DTypeGrad, DType> v = val;
  return 0.5f * op::sqrt(my_pi) * op::exp(v * v) * g;
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_gamma(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  if (type_util::is_same<DTypeGrad, double>::value) {
    return grad * op::gamma(v) * op::special_functions::cephes::psi<double>(v);
  } else {
    return grad * op::gamma(v) * op::special_functions::cephes::psi<float>(v);
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_gammaln(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  if (type_util::is_same<DTypeGrad, double>::value) {
    return grad * op::special_functions::cephes::psi<double>(v);
  } else {
    return grad * op::special_functions::cephes::psi<float>(v);
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_digamma(const DTypeGrad grad, const DType val) {
  const mixed_type<DTypeGrad, DType> v = val;
  if (type_util::is_same<DTypeGrad, double>::value) {
    return grad * op::special_functions::trigamma<double>(v);
  } else {
    return grad * op::special_functions::trigamma<float>(v);
  }
}

template <typename DType, typename DTypeGrad>
__device__ inline mixed_type<DTypeGrad, DType>
backward_gelu(const DTypeGrad grad, const DType val) {
  return 0.5f * (grad + grad * op::erf(val / op::sqrt(2.0f)) +
                 val * backward_erf(grad, val / op::sqrt(2.0f)) / op::sqrt(2.0f));
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

template <typename DType, typename DType2>
__device__ inline DType2 xelu_grad(const DType val,
                                   const DType2 val2) {
  return (val > 0) ? 1 : val2;
}

template <typename DType, typename DType2>
__device__ inline DType prelu_grad(const DType val,
                                   const DType2 val2) {
  return (val > 0) ? 0 : val;
}

}  // namespace op

)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_BACKWARD_FUNCTIONS_INL_H_
