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
 * \file special_functions-inl.h
 * \brief
 * \author Matthias Seeger
*/


#ifndef MXNET_OPERATOR_MATH_FUNCTIONS_INL_H_
#define MXNET_OPERATOR_MATH_FUNCTIONS_INL_H_

#include "math.h"

namespace mxnet {
namespace op {

namespace math {

// Wrappers for math.h unary and binary functions
// - For DType != double: math::name(a) does computation in float
//   and returns float
// - For DType == double: math::name(a) does computation in double
//   and returns double

#define MXNET_UNARY_MATH_FUNC(name) \
template<typename DType> MSHADOW_XINLINE \
float name(DType a) { \
  return ::name##f(static_cast<float>(a)); \
} \
MSHADOW_XINLINE \
double name(double a) { \
  return ::name(a); \
}

#define MXNET_BINARY_MATH_FUNC(name) \
template<typename DType> MSHADOW_XINLINE \
float name(DType a, DType b) { \
  return ::name##f(static_cast<float>(a), static_cast<float>(b)); \
} \
MSHADOW_XINLINE \
double name(double a, double b) { \
  return ::name(a, b); \
}

MXNET_UNARY_MATH_FUNC(exp)

MXNET_UNARY_MATH_FUNC(expm1)

MXNET_UNARY_MATH_FUNC(tanh)

MXNET_UNARY_MATH_FUNC(log1p)

MXNET_UNARY_MATH_FUNC(log)

MXNET_UNARY_MATH_FUNC(log10)

MXNET_UNARY_MATH_FUNC(log2)

MXNET_UNARY_MATH_FUNC(sin)

MXNET_UNARY_MATH_FUNC(cos)

MXNET_UNARY_MATH_FUNC(tan)

MXNET_UNARY_MATH_FUNC(asin)

MXNET_UNARY_MATH_FUNC(sqrt)

MXNET_UNARY_MATH_FUNC(acos)

MXNET_UNARY_MATH_FUNC(atan)

MXNET_UNARY_MATH_FUNC(sinh)

MXNET_UNARY_MATH_FUNC(cosh)

MXNET_UNARY_MATH_FUNC(asinh)

MXNET_UNARY_MATH_FUNC(acosh)

MXNET_UNARY_MATH_FUNC(atanh)

MXNET_UNARY_MATH_FUNC(fabs)

MXNET_UNARY_MATH_FUNC(cbrt)

MXNET_UNARY_MATH_FUNC(round)

MXNET_UNARY_MATH_FUNC(ceil)

MXNET_UNARY_MATH_FUNC(floor)

MXNET_UNARY_MATH_FUNC(trunc)

MXNET_UNARY_MATH_FUNC(tgamma)

MXNET_UNARY_MATH_FUNC(lgamma)

MXNET_BINARY_MATH_FUNC(hypot)

MXNET_BINARY_MATH_FUNC(pow)

template<typename DType> MSHADOW_XINLINE
float id(DType a) {
  return static_cast<float>(a);
}
MSHADOW_XINLINE
double id(double a) {
  return a;
}

template<typename DType> MSHADOW_XINLINE
float sqr(DType a) {
  float af(static_cast<float>(a));
  return af * af;
}
MSHADOW_XINLINE
double sqr(double a) {
  return a * a;
}

}  // namespace math
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MATH_FUNCTIONS_INL_H_
