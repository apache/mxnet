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
 * \file expr_scalar-inl.h
 * \brief definitions of operators in expression with respect to scalar
 *  this file will be included several times, each time with MACRO MSHADOW_SCALAR_ to be different types
 *
 * DO NOT add pragma once or macro guard
 * \author Tianqi Chen, Bing Xu
 */
// macro guard is harmful, used to pass the cpplint
#ifndef MSHADOW_EXPR_SCALAR_INL_H_
#define MSHADOW_EXPR_SCALAR_INL_H_
// undef the guard so it can be included multiple times
#undef MSHADOW_EXPR_SCALAR_INL_H_

namespace mshadow {
namespace expr {
// DotExp
/*! \brief dot operator def */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_>
operator*(const DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_> &lhs,
          MSHADOW_SCALAR_ rhs) {
  return DotExp<TA, TB, ltrans, rtrans,
                MSHADOW_SCALAR_>(lhs.lhs_, lhs.rhs_, lhs.scale_ * rhs);
}
/*! \brief scale of dot operation */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_>
operator*(MSHADOW_SCALAR_ lhs,
          const DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_> &rhs) {
  return DotExp<TA, TB, ltrans, rtrans,
                MSHADOW_SCALAR_>(rhs.lhs_, rhs.rhs_, rhs.scale_ * lhs);
}

/*! \brief operator overload */
template<typename E, typename DType, typename R, int d>
inline ReduceTo1DExp<E, DType, R, d>
operator*(const ReduceTo1DExp<E, DType, R, d> &e, MSHADOW_SCALAR_ scale) {
  return ReduceTo1DExp<E, DType, R, d>(e.src_, e.scale_ * scale);
}
/*! \brief operator overload */
template<typename E, typename DType, typename R, int d>
inline ReduceTo1DExp<E, DType, R, d>
operator*(MSHADOW_SCALAR_ scale, const ReduceTo1DExp<E, DType, R, d> &e) {
  return ReduceTo1DExp<E, DType, R, d>(e.src_, e.scale_ * scale);
}

/*! \brief operator overload for const */
template<typename OP, typename TA, int ta>
inline BinaryMapExp<OP, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
F(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs, const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
/*! \brief operator overload for const */
template<typename OP, typename TB, int tb>
inline BinaryMapExp<OP, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
F(const ScalarExp<MSHADOW_SCALAR_> &lhs, const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
/*! \brief operator overload for const */
template<typename OP>
inline BinaryMapExp<OP, ScalarExp<MSHADOW_SCALAR_>, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (1|type::kMapper)>
F(const ScalarExp<MSHADOW_SCALAR_> &lhs, const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
// constant operators
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::plus, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator+(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::minus, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator-(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::mul, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator*(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::div, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator/(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
// constant operators 2
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::plus, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator+(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::minus, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator-(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::mul, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator*(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::div, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator/(const ScalarExp<MSHADOW_SCALAR_> &lhs, const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
// constant operators 3
/*! \brief operator overload */
inline BinaryMapExp<op::plus, ScalarExp<MSHADOW_SCALAR_>, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (1|type::kMapper)>
operator+(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
inline BinaryMapExp<op::minus, ScalarExp<MSHADOW_SCALAR_>, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (1|type::kMapper)>
operator-(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
inline BinaryMapExp<op::mul, ScalarExp<MSHADOW_SCALAR_>, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (1|type::kMapper)>
operator*(const ScalarExp<MSHADOW_SCALAR_> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
inline BinaryMapExp<op::div, ScalarExp<MSHADOW_SCALAR_>, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (1|type::kMapper)>
operator/(const ScalarExp<MSHADOW_SCALAR_> &lhs, const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPR_SCALAR_INL_H_
