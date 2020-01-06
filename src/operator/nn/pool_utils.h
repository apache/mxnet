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

#ifndef MXNET_OPERATOR_NN_POOL_UTILS_H_
#define MXNET_OPERATOR_NN_POOL_UTILS_H_

#include "../mshadow_op.h"

namespace mxnet {
namespace op {

// Define an accumulator type AccType to permit float16-I/O lp pooling to avoid underflow.
template<typename DType>
struct PoolingTypes {
  typedef DType AccType;
};

template<>
struct PoolingTypes<mshadow::half::half_t> {
  typedef float AccType;
};

template<typename DType, int p>
struct a_pow_p {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return mshadow_op::power::Map(a, DType(p));
  }
};

template<typename DType>
struct a_pow_p<DType, 1> {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return a;
  }
};

template<typename DType>
struct a_pow_p<DType, 2> {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return a*a;
  }
};

template<typename DType>
struct a_pow_p<DType, 3> {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return a*a*a;
  }
};

template<typename DType, int p>
struct a_root_p {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return mshadow_op::power::Map(a, DType(1.0 / p));
  }
};

template<typename DType>
struct a_root_p<DType, 1> {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return a;
  }
};

template<typename DType>
struct a_root_p<DType, 2> {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return mshadow_op::square_root::Map(a);
  }
};

template<typename DType>
struct a_root_p<DType, 3> {
  static MSHADOW_XINLINE DType Map(const DType a) {
    return mshadow_op::cube_root::Map(a);
  }
};

template<typename DType, int p>
struct lp_grad {
  static MSHADOW_XINLINE DType Map(const DType grad, const DType in_data, const DType out_data) {
    return grad * mshadow_op::power::Map(in_data / out_data, DType(p - 1));
  }
};

template<typename DType>
struct lp_grad<DType, 1> {
  static MSHADOW_XINLINE DType Map(const DType grad, const DType in_data, const DType out_data) {
    return grad;
  }
};

template<typename DType>
struct lp_grad<DType, 2> {
  static MSHADOW_XINLINE DType Map(const DType grad, const DType in_data, const DType out_data) {
    // Avoid inf, if out_data has underflowed to 0 for a non-zero input, or nan if grad is also 0.
    return (out_data == DType(0.0)) ? DType(0.0) : grad * (in_data / out_data);
  }
};

template<typename DType>
struct lp_grad<DType, 3> {
  static MSHADOW_XINLINE DType Map(const DType grad, const DType in_data, const DType out_data) {
    // Avoid inf, if out_data has underflowed to 0 for a non-zero input, or nan if grad is also 0.
    DType in_out_ratio = in_data / out_data;
    return (out_data == DType(0.0)) ? DType(0.0) : grad * in_out_ratio * in_out_ratio;
  }
};

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOL_UTILS_H_
