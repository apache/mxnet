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

#ifndef MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
#define MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_

#include <string>

namespace mxnet {

namespace detail {

const std::string fp16_support_string = R"code(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#if defined(__cplusplus)
  struct __align__(2) __half {
    __host__ __device__ __half() { }
  protected:
    unsigned short __x;
  };
  /* All intrinsic functions are only available to nvcc compilers */
  #if defined(__CUDACC__)
    /* Definitions of intrinsics */
    __device__ inline __half __float2half(const float f) {
      __half val;
      asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
      return val;
    }
    __device__ inline float __half2float(const __half h) {
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
      return val;
    }
  #endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */
#undef __HALF_TO_US
#undef __HALF_TO_CUS
typedef __half half;
)code";

const std::string fused_op_function_definitions = R"code(
template <typename DType>
struct LoadType {
  using Type = DType;
};

template <>
struct LoadType<half> {
  using Type = float;
};

template <typename DType>
inline typename LoadType<DType>::Type load(const DType * input, int i) {
  return input[i];
}

template <>
inline float load(const half * input, int i) {
  return __half2float(input[i]);
}

template <typename DType>
inline void store(const typename LoadType<DType>::Type value, int i, DType * output) {
  output[i] = value;
}

template <>
inline void store(const float value, int i, half * output) {
  output[i] = __float2half(value);
}

template <typename DType, typename DType2>
inline DType add(const DType a, const DType2 b) {
  return a + b;
}

template <typename DType, typename DType2>
inline DType sub(const DType a, const DType2 b) {
  return a - b;
}

template <typename DType, typename DType2>
inline DType mul(const DType a, const DType2 b) {
  return a * b;
}

template <typename DType, typename DType2>
inline DType div(const DType a, const DType2 b) {
  return a / b;
}

template <typename DType, typename DType2>
inline DType pow(const DType a, const DType2 b) {
  return powf(a, b);
}

template <typename DType, typename DType2>
inline DType max(const DType a, const DType2 b) {
  return a > b ? a : b;
}

template <typename DType, typename DType2>
inline DType min(const DType a, const DType2 b) {
  return a < b ? a : b;
}

template <typename OutType, typename DType>
inline OutType cast(const DType val) {
  return static_cast<OutType>(val);
}

// activations

template <typename DType>
inline DType relu(const DType val) {
  return val > 0 ? val : 0;
}

template <typename DType>
inline DType backward_relu(const DType val, const DType grad) {
  return val > 0 ? grad : 0;
}

template <typename DType>
inline DType sigmoid(const DType val) {
  return 1.f/(1 + expf(-val));
}

template <typename DType>
inline DType backward_sigmoid(const DType val, const DType grad) {
  const DType ep1 = 1 + expf(-val);
  return grad * expf(-val)/(ep1*ep1);
}

template <typename DType>
inline DType softrelu(const DType val) {
  return logf(1 + expf(val));
}

template <typename DType>
inline DType backward_softrelu(const DType val, const DType grad) {
  return grad * sigmoid(val);
}

template <typename DType>
inline DType softsign(const DType val) {
  return val / (1 + absf(val));
}

template <typename DType>
inline DType backward_softsign(const DType val, const DType grad) {
  const DType ap1 = 1 + absf(val);
  return grad / (ap1 * ap1);
}

// exp and log

template <typename DType>
inline DType exp(const DType val) {
  return expf(val);
}

template <typename DType>
inline DType backward_exp(const DType val, const DType grad) {
  return grad * expf(val);
}

template <typename DType>
inline DType expm1(const DType val) {
  return expm1f(val);
}

template <typename DType>
inline DType backward_expm1(const DType val, const DType grad) {
  return grad * expf(val);
}

template <typename DType>
inline DType log(const DType val) {
  return logf(val);
}

template <typename DType>
inline DType backward_log(const DType val, const DType grad) {
  return grad / val;
}

template <typename DType>
inline DType log10(const DType val) {
  return log10f(val);
}

template <typename DType>
inline DType backward_log10(const DType val, const DType grad) {
  return grad / (val * logf(10));
}

template <typename DType>
inline DType log2(const DType val) {
  return log2f(val);
}

template <typename DType>
inline DType backward_log2(const DType val, const DType grad) {
  return grad / (val * logf(2));
}

template <typename DType>
inline DType log1p(const DType val) {
  return log1pf(val);
}

template <typename DType>
inline DType backward_log1p(const DType val, const DType grad) {
  return grad / (1 + val);
}

// trigonometric

template <typename DType>
inline DType sin(const DType val) {
  return sinf(val);
}

template <typename DType>
inline DType backward_sin(const DType val, const DType grad) {
  return grad * cosf(val);
}

template <typename DType>
inline DType cos(const DType val) {
  return cosf(val);
}

template <typename DType>
inline DType backward_cos(const DType val, const DType grad) {
  return -grad * sinf(val);
}

template <typename DType>
inline DType tan(const DType val) {
  return tanf(val);
}

// Uses output from tan
template <typename DType>
inline DType backward_tan(const DType val, const DType grad) {
  return grad * (val * val + 1);
}

template <typename DType>
inline DType arcsin(const DType val) {
  return asinf(val);
}

template <typename DType>
inline DType backward_arcsin(const DType val, const DType grad) {
  return grad / sqrtf(1 - val*val);
}

template <typename DType>
inline DType arccos(const DType val) {
  return acosf(val);
}

template <typename DType>
inline DType backward_arccos(const DType val, const DType grad) {
  return -grad / sqrtf(1 - val*val);
}

template <typename DType>
inline DType arctan(const DType val) {
  return atanf(val);
}

template <typename DType>
inline DType backward_arctan(const DType val, const DType grad) {
  return grad / (1 + val*val);
}

template <typename DType>
inline DType sinh(const DType val) {
  return sinhf(val);
}

template <typename DType>
inline DType backward_sinh(const DType val, const DType grad) {
  return grad * coshf(val);
}

template <typename DType>
inline DType cosh(const DType val) {
  return coshf(val);
}

template <typename DType>
inline DType backward_cosh(const DType val, const DType grad) {
  return grad * sinhf(val);
}

template <typename DType>
inline DType tanh(const DType val) {
  return tanhf(val);
}

// Uses tanh output
template <typename DType>
inline DType backward_tanh(const DType val, const DType grad) {
  return grad * (1 - val * val);
}

template <typename DType>
inline DType arcsinh(const DType val) {
  return asinhf(val);
}

template <typename DType>
inline DType backward_arcsinh(const DType val, const DType grad) {
  return grad / sqrtf(val * val + 1);
}

template <typename DType>
inline DType arccosh(const DType val) {
  return acoshf(val);
}

template <typename DType>
inline DType backward_arccosh(const DType val, const DType grad) {
  return grad / sqrtf(val * val - 1);
}

template <typename DType>
inline DType arctanh(const DType val) {
  return atanhf(val);
}

template <typename DType>
inline DType backward_arctanh(const DType val, const DType grad) {
  return grad / (1 - val * val);
}

// sqrt

template <typename DType>
inline DType sqrt(const DType val) {
  return sqrtf(val);
}

template <typename DType>
inline DType backward_sqrt(const DType val, const DType grad) {
  return 0.5 * grad * rsqrtf(val);
}

template <typename DType>
inline DType rsqrt(const DType val) {
  return rsqrtf(val);
}

template <typename DType>
inline DType backward_rsqrt(const DType val, const DType grad) {
  const DType inv = 1 / val;
  return -0.5 * grad * sqrtf(inv) * inv;
}

template <typename DType>
inline DType cbrt(const DType val) {
  return cbrtf(val);
}

template <typename DType>
inline DType backward_cbrt(const DType val, const DType grad) {
  const DType inv = rcbrtf(val);
  return 1.f/3.f * grad * inv * inv;
}

template <typename DType>
inline DType rcbrt(const DType val) {
  return rcbrtf(val);
}

template <typename DType>
inline DType backward_rcbrt(const DType val, const DType grad) {
  const DType inv = 1 / val;
  return -1.f/3.f * grad * cbrtf(inv) * inv;
}

)code";

const std::string fused_op_kernel_begin = R"code(
const int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < N; i+= gridDim.x * blockDim.x) {
)code";

const std::string fused_op_kernel_end = R"code(
}
}
)code";

}  // namespace detail

}  // namespace mxnet

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
