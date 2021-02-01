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

static_assert(sizeof(float32) == 4, "Size of float32 is expected to be 4B");
static_assert(sizeof(float64) == 8, "Size of float64 is expected to be 8B");
static_assert(sizeof(float16) == 2, "Size of float16 is expected to be 2B");
static_assert(sizeof(uint8) == 1, "Size of uint8 is expected to be 1B");
static_assert(sizeof(int8) == 1, "Size of int8 is expected to be 1B");
static_assert(sizeof(int32) == 4, "Size of int32 is expected to be 4B");
static_assert(sizeof(int64) == 8, "Size of int64 is expected to be 8B");

)code"
#if MSHADOW_INT64_TENSOR_SIZE == 1
"typedef int64 index_t;\n"
#else
"typedef int32 index_t;\n"
#endif
R"code(
// bool and int8 need to be accumulated in index_t
// but bool needs to be treated in the special way
// for ops like bitwise_not
struct bool_t {
  index_t value;

  __device__ inline bool_t(const index_t& v) : value(v) {}
  __device__ inline bool_t(const volatile index_t& v) : value(v) {}
  __device__ inline bool_t() : value(0) {}

  __device__ inline operator index_t() const volatile { return value; }
  __device__ inline bool_t& operator= (const index_t& v) {
    value = v;
    return *this;
  }
  __device__ inline volatile bool_t& operator= (const index_t& v) volatile {
    value = v;
    return *this;
  }
  __device__ inline bool_t& operator= (const volatile index_t& v) {
    value = v;
    return *this;
  }
};
template<>
struct AccType<bool> {
  using type = bool_t;

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
template <> struct is_integral<bool_t>  : true_type {};

// is_unsigned
template <typename T> struct is_unsigned : false_type {};
template <> struct is_unsigned<uint8> : true_type {};
template <> struct is_unsigned<bool>  : true_type {};
template <> struct is_unsigned<bool_t>  : true_type {};

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
struct mixed_type_helper;

template <typename T>
struct mixed_type_helper<T, float64, typename enable_if<!is_same<float64, T>::value>::type> {
  using type = float64;
};

template <typename T>
struct mixed_type_helper<float64, T> {
  using type = float64;
};

template <typename T>
struct mixed_type_helper<T, float32, typename enable_if<!is_same<float64, T>::value &&
                                                        !is_same<float32, T>::value>::type> {
  using type = float32;
};

template <typename T>
struct mixed_type_helper<float32, T, typename enable_if<!is_same<float64, T>::value>::type> {
  using type = float32;
};

template <typename T>
struct mixed_type_helper<T, float16, typename enable_if<is_same<float16, T>::value ||
                                                        is_integral<T>::value>::type> {
  using type = float16;
};

template <typename T>
struct mixed_type_helper<float16, T, typename enable_if<is_integral<T>::value>::type> {
  using type = float16;
};

template <typename T, typename U>
struct mixed_type_helper<T, U, typename enable_if<is_integral<T>::value &&
                                                  is_integral<U>::value &&
                                                  !is_same<U, bool_t>::value &&
                                                  sizeof(T) <= sizeof(U)>::type> {
  using type = U;
};

template <typename T, typename U>
struct mixed_type_helper<U, T, typename enable_if<is_integral<T>::value &&
                                                  is_integral<U>::value &&
                                                  !is_same<U, bool_t>::value &&
                                                  sizeof(T) < sizeof(U)>::type> {
  using type = U;
};

template <typename T>
struct mixed_type_helper<T, bool_t, typename enable_if<is_integral<T>::value &&
                                                       sizeof(T) < sizeof(bool_t)>::type> {
  using type = index_t;
};

template <typename T>
struct mixed_type_helper<bool_t, T, typename enable_if<is_integral<T>::value &&
                                                       sizeof(T) < sizeof(bool_t)>::type> {
  using type = index_t;
};

template <typename T>
struct mixed_type_helper<T, bool_t, typename enable_if<is_integral<T>::value &&
                                                       sizeof(T) == sizeof(bool_t)>::type> {
  using type = T;
};

template <typename... Ts>
struct multi_mixed_type_helper;

template <>
struct multi_mixed_type_helper<> {
    using type = void;
};

template <typename T>
struct multi_mixed_type_helper<T> {
    using type = T;
};

template <typename T, typename U, typename... Ts>
struct multi_mixed_type_helper<T, U, Ts...> {
    using type = typename mixed_type_helper<T,
                                            typename multi_mixed_type_helper<U,
                                                                             Ts...>::type>::type;
};

template <typename... Ts>
using mixed_type = typename multi_mixed_type_helper<Ts...>::type;

}  // namespace type_util
)code";

const char util_string[] = R"code(
enum class OpReqType {
  kNullOp,
  kWriteTo,
  kWriteInplace,
  kAddTo
};

constexpr int kRTCMaxThreadsPerBlock = 512;
constexpr int warp_size = 32;

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

template <int NVALUES = warp_size, typename OP, typename T>
__device__ inline T warp_reduce(T value, OP redfun) {
#pragma unroll
  for (int i = warp_size / 2; i >= 1; i /= 2) {
    if (NVALUES > i) value = redfun(value, __shfl_down_sync(0xffffffff, value, i));
  }
  return value;
}

template <typename OP, typename T>
__device__ inline T grouped_warp_reduce(T value, OP redfun, const int group_size) {
  for (int i = 1; i < group_size; i *= 2) {
    value = redfun(value, __shfl_down_sync(0xffffffff, value, i));
  }
  return value;
}

template <typename OP, typename T>
__device__ inline T grouped_warp_allreduce(T value, OP redfun, const int group_size) {
  value = grouped_warp_reduce(value, redfun, group_size);
  return __shfl_sync(0xffffffff, value, 0, group_size);
}

template <typename OP, typename T>
__device__ inline T strided_grouped_warp_reduce(T value, OP redfun, const int group_size) {
  for (int i = warp_size / 2; i >= group_size; i /= 2) {
    value = redfun(value, __shfl_down_sync(0xffffffff, value, i));
  }
  return value;
}

template <typename OP, typename T>
__device__ inline T strided_grouped_warp_allreduce(T value, OP redfun, const int group_size) {
  value = strided_grouped_warp_reduce(value, redfun, group_size);
  for (int i = group_size; i < warp_size; i *= 2) {
    T tmp = __shfl_up_sync(0xffffffff, value, i);
    if (threadIdx.x % warp_size >= i) {
      value = tmp;
    }
  }
  return value;
}

}  // namespace util
)code";
}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_UTIL_INL_H_
