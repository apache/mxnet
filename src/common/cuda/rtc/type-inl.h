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

#ifndef MXNET_COMMON_CUDA_RTC_TYPE_INL_H_
#define MXNET_COMMON_CUDA_RTC_TYPE_INL_H_

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


}  // namespace type_util
)code"
#if MSHADOW_INT64_TENSOR_SIZE == 1
"typedef int64 index_t;\n";
#else
"typedef int32 index_t;\n";
#endif

const char op_req_type_string[] = R"code(
enum class OpReqType {
  kNullOp,
  kWriteTo,
  kWriteInplace,
  kAddTo
};
)code";
}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_TYPE_INL_H_
