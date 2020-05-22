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

#ifndef MXNET_COMMON_CUDA_RTC_FORWARD_FUNCTIONS_INL_H_
#define MXNET_COMMON_CUDA_RTC_FORWARD_FUNCTIONS_INL_H_

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char function_definitions[] = R"code(

#define INT_MAX (2147483647)

namespace op {

template <typename DType>
struct LoadType {
  using Type = DType;
};

template <>
struct LoadType<half> {
  using Type = float;
};

template <typename DType>
__device__ inline typename LoadType<DType>::Type load(const DType input) {
  return input;
}

template <>
__device__ inline float load(const half input) {
  return __half2float(input);
}

template <typename DType1, typename DType2>
__device__ inline DType1 store(const DType2 input, DType1* ref) {
  return input;
}

template <typename DType>
__device__ inline half store(const DType input, half* ref) {
  return __float2half(input);
}

template <int size>
struct VectorConfig {
    static_assert(size >= 4, "VectorConfig needs to have size of at least 4B");
    using IndexType = float;
};

template <>
struct VectorConfig<8> {
    using IndexType = double;
};

template <>
struct VectorConfig<16> {
    using IndexType = double2;
};

template <>
struct VectorConfig<32> {
    using IndexType = double4;
};

template <typename DType>
__device__ inline DType add_elem(const DType& x, const DType& y) {
  return x + y;
}

template <>
__device__ inline half add_elem(const half& x, const half& y) {
  return __float2half(__half2float(x) + __half2float(y));
}

template <typename DType, int nvec>
union VectorType {
    typename VectorConfig<sizeof(DType)*nvec>::IndexType y;
    DType x[nvec];
    __device__ VectorType () {};
    __device__ VectorType (const VectorType<DType, nvec>& y2) {
        y = y2.y;
    }
    __device__ VectorType (const decltype(y) &y2) {
        y = y2;
    }
    __device__ inline VectorType<DType, nvec>& operator+=(const VectorType<DType, nvec>& rhs) {
      #pragma unroll
      for (int i = 0; i < nvec; ++i) {
        x[i] = add_elem(x[i], rhs.x[i]);
      }
      return *this;
    }
};

template <int ndim>
struct Shape {
   int x[ndim];
   size_t size;
   __device__ inline const int& operator [](const int i) const {
       return x[i];
   }
   __device__ inline int& operator [](const int i) {
       return x[i];
   }
   __device__ inline void set(const int def) {
       #pragma unroll
       for (int i = 0; i < ndim; i++) {
           x[i] = def;
       }
   }
};

template <>
struct Shape<0> {
   size_t size;
};

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> load_index(const DType * input, int i,
                                                     const Shape<ndim> &shape) {
  if (i < shape.size) {
    const auto* vector_input = reinterpret_cast<
                                const typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(
                                    input + i);
    VectorType<DType, nvec> ret = {*vector_input};
    return ret;
  } else {
    VectorType<DType, nvec> ret({0});
    return ret;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> global_load_index(const DType * input, int i,
                                                            const Shape<ndim> &shape) {
  if (i < shape.size) {
    const auto* vector_input = reinterpret_cast<
                                const typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(
                                    input + i);
    VectorType<DType, nvec> ret = {__ldg(vector_input)};
    return ret;
  } else {
    VectorType<DType, nvec> ret({0});
    return ret;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> load_slice(const DType * input, const Shape<ndim>& shape,
                                                     Shape<ndim> begin, Shape<ndim> end,
                                                     int offset) {
  int idx[nvec];

  Shape<ndim> ref_strides;
  Shape<ndim> strides;
  ref_strides[ndim-1] = 1;
  strides[ndim-1] = 1;
  #pragma unroll
  for (int dim = ndim-1; dim >=0; dim--) {
    if (begin[dim] < 0) begin[dim] = shape[dim] + begin[dim];
    if (end[dim] < 0) end[dim] = shape[dim] + end[dim];
    if (end[dim] == INT_MAX) end[dim] = shape[dim];
    if (dim > 0) {
      ref_strides[dim-1] = ref_strides[dim] * (end[dim] - begin[dim]);
      strides[dim-1] = strides[dim] * shape[dim];
    }
  }
  #pragma unroll
  for (int j = 0; j < nvec; j++) {
    idx[j] = 0;
    int ref_idx = offset + j;
    #pragma unroll
    for (int dim = 0; dim < ndim; dim++) {
       int stride = ref_strides[dim];
       if (shape[dim] > 1) {
         idx[j] += (ref_idx / stride + begin[dim]) * strides[dim];
       }
       ref_idx = ref_idx % stride;
    }
  }
  VectorType<DType, nvec> ret;
  #pragma unroll
  for (int j = 0; j < nvec; j++) {
      ret.x[j] = *(input + idx[j]);
  }
  return ret;
}

template <int nvec, typename DType, int ndim>
__device__ inline VectorType<DType, nvec> fast_load_slice(const DType * input,
                                                          const Shape<ndim>& shape,
                                                          Shape<ndim> begin,
                                                          Shape<ndim> end,
                                                          int offset) {
  int idx = 0;

  Shape<ndim> ref_strides;
  Shape<ndim> strides;
  ref_strides[ndim-1] = 1;
  strides[ndim-1] = 1;
  #pragma unroll
  for (int dim = ndim-1; dim >=0; dim--) {
    if (begin[dim] < 0) begin[dim] = shape[dim] + begin[dim];
    if (end[dim] < 0) end[dim] = shape[dim] + end[dim];
    if (end[dim] == INT_MAX) end[dim] = shape[dim];
    if (dim > 0) {
      ref_strides[dim-1] = ref_strides[dim] * (end[dim] - begin[dim]);
      strides[dim-1] = strides[dim] * shape[dim];
    }
  }
  int ref_idx = offset;
  #pragma unroll
  for (int dim = 0; dim < ndim; dim++) {
     int stride = ref_strides[dim];
     if (shape[dim] > 1) {
       idx += (ref_idx / stride + begin[dim]) * strides[dim];
     }
     ref_idx = ref_idx % stride;
  }
  return global_load_index<nvec>(input, idx, shape);
}

template <int nvec, typename DType, int ndim>
__device__ inline void store_index(const VectorType<DType, nvec> value, int i,
                        DType * output, const Shape<ndim>& shape) {
  if (i < (shape.size + nvec - 1) / nvec) {
    auto vector_output = reinterpret_cast<
                          typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(output);
    vector_output[i] = value.y;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline void store_add_index(const VectorType<DType, nvec> value, int i,
                            DType * output, const Shape<ndim>& shape) {
  if (i < (shape.size + nvec - 1) / nvec) {
    auto vector_output = reinterpret_cast<
                          typename VectorConfig<sizeof(DType)*nvec>::IndexType *>(output);
    VectorType<DType, nvec> ret(vector_output[i]);
    ret += value;
    vector_output[i] = ret.y;
  }
}

template <typename DType>
__device__ inline DType identity(const DType val) {
  return val;
}

template <typename DType>
__device__ inline DType negation(const DType val) {
  return -val;
}

template <typename DType, typename DType2>
__device__ inline DType add(const DType a, const DType2 b) {
  return a + b;
}

template <typename DType, typename DType2>
__device__ inline DType sub(const DType a, const DType2 b) {
  return a - b;
}

template <typename DType, typename DType2>
__device__ inline DType mul(const DType a, const DType2 b) {
  return a * b;
}

template <typename DType, typename DType2>
__device__ inline DType div(const DType a, const DType2 b) {
  return a / b;
}

template <typename DType, typename DType2>
__device__ inline DType rdiv(const DType a, const DType2 b) {
  return b / a;
}

#define DEFINE_BINARY_MATH_FUNC(name, double_version, float_version) \
template <typename DType, typename DType2> \
__device__ inline DType name (const DType a, const DType2 b) { \
  if (type_util::has_double_or_integral<DType, DType2>::value) { \
    return double_version ((double)a, (double)b); \
  } else { \
    return float_version (a, b); \
  } \
}

DEFINE_BINARY_MATH_FUNC(power, ::pow, ::powf)

template <typename DType, typename DType2>
__device__ inline DType rpow(const DType a, const DType2 b) {
  return power(b, a);
}

template <typename DType, typename DType2>
__device__ inline DType max(const DType a, const DType2 b) {
  if (isnan(a)) return a;
  return a > b ? a : b;
}

template <typename DType, typename DType2>
__device__ inline DType min(const DType a, const DType2 b) {
  if (isnan(a)) return a;
  return a < b ? a : b;
}

DEFINE_BINARY_MATH_FUNC(hypot, ::hypot, ::hypotf)

#undef DEFINE_BINARY_MATH_FUNC

template <typename OutType, typename DType>
__device__ inline typename LoadType<OutType>::Type cast(const DType val) {
  return static_cast<typename LoadType<OutType>::Type>(val);
}

// activations

template <typename DType>
__device__ inline DType relu(const DType val) {
  return (isnan(val) || val > 0) ? val : 0;
}

template <typename DType>
__device__ inline DType sigmoid(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return 1./(1 + ::exp(-val));
  } else {
    return 1.f/(1 + expf(-val));
  }
}

template <typename DType>
__device__ inline DType softrelu(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return ::log(1 + ::exp(val));
  } else {
    return logf(1 + expf(val));
  }
}

template <typename DType>
__device__ inline DType softsign(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return val / (1 + fabs(val));
  } else {
    return val / (1 + fabsf(val));
  }
}

// exp and log

#define DEFINE_UNARY_MATH_FUNC(name, double_version, float_version) \
template <typename DType> \
__device__ inline DType name (const DType a) { \
  if (type_util::has_double_or_integral<DType>::value) { \
    return double_version ((double)a); \
  } else { \
    return float_version (a); \
  } \
}

DEFINE_UNARY_MATH_FUNC(exp, ::exp, ::expf)
DEFINE_UNARY_MATH_FUNC(expm1, ::expm1, ::expm1f)
DEFINE_UNARY_MATH_FUNC(log, ::log, ::logf)
DEFINE_UNARY_MATH_FUNC(log10, ::log10, ::log10f)
DEFINE_UNARY_MATH_FUNC(log2, ::log2, ::log2f)
DEFINE_UNARY_MATH_FUNC(log1p, ::log1p, ::log1pf)

// trigonometric

constexpr double pi = 3.14159265358979323846;

template <typename DType>
__device__ inline DType degrees(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return (val / pi) * 180;
  } else {
    return (val / static_cast<float>(pi)) * 180.f;
  }
}

template <typename DType>
__device__ inline DType radians(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return (val / 180.0) * pi;
  } else {
    return (val / 180.0f) * static_cast<float>(pi);
  }
}

DEFINE_UNARY_MATH_FUNC(sin, ::sin, ::sinf)
DEFINE_UNARY_MATH_FUNC(cos, ::cos, ::cosf)
DEFINE_UNARY_MATH_FUNC(tan, ::tan, ::tanf)
DEFINE_UNARY_MATH_FUNC(arcsin, ::asin, ::asinf)
DEFINE_UNARY_MATH_FUNC(arccos, ::acos, ::acosf)
DEFINE_UNARY_MATH_FUNC(arctan, ::atan, ::atanf)

DEFINE_UNARY_MATH_FUNC(sinh, ::sinh, ::sinhf)
DEFINE_UNARY_MATH_FUNC(cosh, ::cosh, ::coshf)
DEFINE_UNARY_MATH_FUNC(tanh, ::tanh, ::tanhf)
DEFINE_UNARY_MATH_FUNC(arcsinh, ::asinh, ::asinhf)
DEFINE_UNARY_MATH_FUNC(arccosh, ::acosh, ::acoshf)
DEFINE_UNARY_MATH_FUNC(arctanh, ::atanh, ::atanhf)

// sqrt

DEFINE_UNARY_MATH_FUNC(sqrt, ::sqrt, ::sqrtf)
DEFINE_UNARY_MATH_FUNC(rsqrt, ::rsqrt, ::rsqrtf)
DEFINE_UNARY_MATH_FUNC(cbrt, ::cbrt, ::cbrtf)
DEFINE_UNARY_MATH_FUNC(rcbrt, ::rcbrt, ::rcbrtf)

template <typename DType>
__device__ inline DType square(const DType val) {
  return val * val;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type zero(const DType val) {
  return 0;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type zero() {
  return 0;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type one(const DType val) {
  return 1;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type one() {
  return 1;
}

template <typename DType>
__device__ inline DType round(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::round((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::roundf(val);
  }
}

template <typename DType>
__device__ inline DType floor(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::floor((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::floorf(val);
  }
}

template <typename DType>
__device__ inline DType ceil(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::ceil((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::ceilf(val);
  }
}

template <typename DType>
__device__ inline DType rint(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::rint((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::rintf(val);
  }
}

template <typename DType>
__device__ inline DType fix(const DType val) {
  const auto f = floor(val);
  const auto c = ceil(val);
  return (f > 0 ? f : -f) < (c > 0 ? c : -c) ? f : c;
}

template <typename DType>
__device__ inline DType trunc(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::trunc((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::truncf(val);
  }
}

template <typename DType>
__device__ inline DType clip(const DType val, const float a_min, const float a_max) {
  return max(min(val, a_max), a_min);
}

template <typename DType>
__device__ inline DType sign(const DType val) {
  if (val < 0) return -1;
  return val > 0 ? 1 : 0;
}

template <typename DType>
__device__ inline DType reciprocal(const DType val) {
  return 1.0f / val;
}

DEFINE_UNARY_MATH_FUNC(abs, ::fabs, ::fabsf)
DEFINE_UNARY_MATH_FUNC(gamma, ::tgamma, ::tgammaf)
DEFINE_UNARY_MATH_FUNC(gammaln, ::lgamma, ::lgammaf)
DEFINE_UNARY_MATH_FUNC(erf, ::erf, ::erff)
DEFINE_UNARY_MATH_FUNC(erfinv, ::erfinv, ::erfinvf)

template <typename DType1, typename DType2>
__device__ inline DType1 smooth_l1(const DType1 val, const DType2 scalar) {
  const auto bsq = scalar * scalar;
  const auto ibsq = 1.0f / bsq;
  if (val > ibsq) {
    return val - 0.5f * ibsq;
  } else if (val < -ibsq) {
    return -val - 0.5f * ibsq;
  } else {
    return 0.5f * val * val * bsq;
  }
}

template <typename DType>
__device__ inline DType logical_not(const DType val) {
  return val != DType(0) ? DType(0) : DType(1);
}

template <typename DType>
__device__ inline bool np_logical_not(const DType val) {
  return !static_cast<bool>(val);
}

#undef DEFINE_UNARY_MATH_FUNC

}  // namespace op

)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_FORWARD_FUNCTIONS_INL_H_
