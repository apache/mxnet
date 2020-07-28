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

#ifndef MXNET_COMMON_CUDA_RTC_UTIL_INL_H_
#define MXNET_COMMON_CUDA_RTC_UTIL_INL_H_

#include <mxnet/base.h>

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char type_support_string[] = R"code(
using float32 = float;
using float64 = double;
using float16 = half;
using uint8 = unsigned char;
using int8 = char;
using int32 = int;
using int64 = long long;
)code"
#if MSHADOW_INT64_TENSOR_SIZE == 1
"typedef int64 index_t;\n"
#else
"typedef int32 index_t;\n"
#endif
R"code(
// bool and int8 need to be accumulated in index_t
template<>
struct AccType<bool> {
  using type = index_t;

  __device__ static inline type from(const bool& val) {
    return val;
  }

  __device__ static inline bool to(type val) {
    return val;
  }
};

template<>
struct AccType<int8> {
  using type = index_t;

  __device__ static inline type from(const int8& val) {
    return val;
  }

  __device__ static inline int8 to(type val) {
    return val;
  }
};

template<>
struct AccType<uint8> {
  using type = index_t;

  __device__ static inline type from(const uint8& val) {
    return val;
  }

  __device__ static inline uint8 to(type val) {
    return val;
  }
};

namespace type_util {

struct false_type {
  static constexpr bool value = false;
};

struct true_type {
  static constexpr bool value = true;
};

// is_integral
template <typename T> struct is_integral : false_type {};
template <> struct is_integral<uint8> : true_type {};
template <> struct is_integral<int8>  : true_type {};
template <> struct is_integral<int32> : true_type {};
template <> struct is_integral<int64> : true_type {};
template <> struct is_integral<bool>  : true_type {};

// is_unsigned
template <typename T> struct is_unsigned : false_type {};
template <> struct is_unsigned<uint8> : true_type {};
template <> struct is_unsigned<bool>  : true_type {};

// is_same
template <typename T, typename U>
struct is_same : false_type {};
template <typename T> struct is_same<T, T> : true_type {};

// has_double
template <typename... T> struct has_double : false_type {};

template <typename A, typename... B>
struct has_double<A, B...> {
    static constexpr bool value = is_same<A, double>::value ||
                                  has_double<B...>::value;
};

// has_double_or_integral
template <typename... T> struct has_double_or_integral : false_type {};

template <typename A, typename... B>
struct has_double_or_integral<A, B...> {
    static constexpr bool value = is_same<A, double>::value ||
                                  is_integral<A>::value ||
                                  has_double_or_integral<B...>::value;
};

template <bool b>
struct enable_if {};

template <>
struct enable_if<true> {
  using type = void;
};

template <typename T, typename U, class Enable = void>
struct mixed_type;

template <typename T>
struct mixed_type<T, float64, typename enable_if<!is_same<float64, T>::value>::type> {
  using type = float64;
};

template <typename T>
struct mixed_type<float64, T> {
  using type = float64;
};

template <typename T>
struct mixed_type<T, float32, typename enable_if<!is_same<float64, T>::value &&
                                                 !is_same<float32, T>::value>::type> {
  using type = float32;
};

template <typename T>
struct mixed_type<float32, T, typename enable_if<!is_same<float64, T>::value>::type> {
  using type = float32;
};

template <typename T>
struct mixed_type<T, float16, typename enable_if<is_same<float16, T>::value ||
                                                 is_integral<T>::value>::type> {
  using type = float16;
};

template <typename T>
struct mixed_type<float16, T, typename enable_if<is_integral<T>::value>::type> {
  using type = float16;
};

template <typename T, typename U>
struct mixed_type<T, U, typename enable_if<is_integral<T>::value &&
                                           is_integral<U>::value &&
                                           sizeof(T) <= sizeof(U)>::type> {
  using type = U;
};

template <typename T, typename U>
struct mixed_type<U, T, typename enable_if<is_integral<T>::value &&
                                           is_integral<U>::value &&
                                           sizeof(T) < sizeof(U)>::type> {
  using type = U;
};

}  // namespace type_util
)code";

const char util_string[] = R"code(
enum class OpReqType {
  kNullOp,
  kWriteTo,
  kWriteInplace,
  kAddTo
};

namespace util {

constexpr int MAX_DIM = 5;

template <int ndim>
__device__ inline void unravel_dot(const index_t idx, const index_t (&shape)[MAX_DIM],
  const index_t (&stridej)[MAX_DIM], const index_t (&stridek)[MAX_DIM], index_t* j, index_t* k) {
  *j = 0;
  *k = 0;
  #pragma unroll
  for (index_t i = ndim-1, idx_t = idx; i >=0; --i) {
    const auto tmp = idx_t / shape[i];
    const auto coord = idx_t - tmp*shape[i];
    *j += coord*stridej[i];
    *k += coord*stridek[i];
    idx_t = tmp;
  }
}

template<int ndim>
__device__ inline index_t unravel_dot(const index_t idx, const index_t (&shape)[MAX_DIM],
  const index_t (&stride)[MAX_DIM]) {
  index_t ret = 0;
  #pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    ret += (j - tmp*shape[i])*stride[i];
    j = tmp;
  }
  return ret;
}

template<int ndim>
__device__ inline index_t unravel_ravel(const index_t idx, const index_t (&shape1)[MAX_DIM],
                                        const index_t (&shape2)[MAX_DIM]) {
  index_t ret = 0;
  index_t total_shape = 1;
#pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    if (i != ndim - 1) {
      total_shape *= shape2[i + 1];
    }
    auto tmp = j / shape1[i];
    const index_t coord = j - tmp*shape1[i];
    ret += total_shape * (shape2[i] > coord) * coord;
    j = tmp;
  }
  return ret;
}

template<int ndim, int ndim2>
__device__ inline index_t ravel(const index_t (&coord)[ndim], const index_t (&shape)[ndim2]) {
  index_t ret = 0;
#pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > coord[i]) * coord[i];
  }
  return ret;
}

template<int ndim, int ndim2>
__device__ inline void unravel(const index_t idx,
                               const index_t (&shape)[ndim2],
                               index_t (&coord)[ndim]) {
#pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    coord[i] = j - tmp*shape[i];
    j = tmp;
  }
}

template <typename DType>
__device__ inline bool isinf(volatile const DType &val) {
  return false;
}

template <>
__device__ inline bool isinf(volatile const float &val) {
  return ::isinf(val);
}

template <>
__device__ inline bool isinf(volatile const double &val) {
  return ::isinf(val);
}

template <>
__device__ inline bool isinf(volatile const long double &val) {
  return ::isinf(val);
}

template <>
__device__ inline bool isinf(volatile const float16 &val) {
  return ::isinf(__half2float(const_cast<const float16&>(val)));
}

template <typename DType>
__device__ inline bool isnan(volatile const DType &val) {
  return false;
}

template <>
__device__ inline bool isnan(volatile const float &val) {
  return ::isnan(val);
}

template <>
__device__ inline bool isnan(volatile const double &val) {
  return ::isnan(val);
}

template <>
__device__ inline bool isnan(volatile const long double &val) {
  return ::isnan(val);
}

template <>
__device__ inline bool isnan(volatile const float16 &val) {
  return ::isnan(__half2float(const_cast<const float16&>(val)));
}

}  // namespace util
)code";
}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_UTIL_INL_H_
