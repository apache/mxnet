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
 * \file expr_operator.h
 * \brief Common operators defined for Expr.
 *
 * \note Most of the operator defined here perform simple constant folding
 *   when the type is int32 or int64 for simplifying the index expressions.
 */
// Acknowledgement: This file originates from incubator-tvm
// Acknowledgement: Most operator APIs originate from Halide.
#ifndef MXNET_EXPR_OPERATOR_H_
#define MXNET_EXPR_OPERATOR_H_

#include <mxnet/ir/expr.h>

namespace mxnet {

template<typename ValueType>
inline PrimExpr MakeConstScalar(MXNetDataType t, ValueType value) {
  if (t.is_int()) return IntImm(t, static_cast<int64_t>(value));
  if (t.is_float()) return FloatImm(t, static_cast<double>(value));
  // customized type and uint is not supported for MXNet for now
  LOG(FATAL) << "cannot make const for type " << t;
  return PrimExpr();
}


template<typename ValueType>
inline PrimExpr make_const(MXNetDataType t, ValueType value) {
  if (t.lanes() == 1) {
    return MakeConstScalar(t, value);
  } else {
    LOG(FATAL) << "MXNetDataType::lanes() != 1 is not supported ";
  }
  return PrimExpr();
}

}  // namespace mxnet

#endif  // MXNET_EXPR_OPERATOR_H_
