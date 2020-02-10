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
 * Copyright (c) 2020 by Contributors
 * \file mxnet_op_kernel_assign.h
 * \brief move kernel assign code from mxnet_op.h to avoid WIN 
 *        compiler fail:compiler is out of heap space in pass 2
 */
#ifndef MXNET_OPERATOR_MXNET_OP_KERNEL_ASSIGN_H_
#define MXNET_OPERATOR_MXNET_OP_KERNEL_ASSIGN_H_

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>

namespace mxnet {
namespace op {
namespace mxnet_op {
using namespace mshadow;

/*!
 * \brief assign the val to out according
 * to request in Kernel::Launch
 * \param out the data to be assigned
 * \param req the assignment request
 * \param val the value to be assigned to out
 * \tparam OType output type
 * \tparam VType value type
 */
#define KERNEL_ASSIGN(out, req, val)  \
  {                                   \
    switch (req) {                    \
      case kNullOp:                   \
        break;                        \
      case kWriteTo:                  \
      case kWriteInplace:             \
        (out) = (val);                \
        break;                        \
      case kAddTo:                    \
        (out) += (val);               \
        break;                        \
      default:                        \
        break;                        \
    }                                 \
  }

/*! \brief Select assignment operation based upon the req value
 * Also useful for mapping mshadow Compute (F<OP>) to Kernel<OP>::Launch
 */
template<typename OP, int req>
struct op_with_req {
  typedef OP Operation;

  /*! \brief input is one tensor */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i]));
  }

  /*! \brief inputs are two tensors */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is tensor and a scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }

  /*! \brief input is tensor and two scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *in,
                                  const DType value_1, const DType value_2) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value_1, value_2));
  }

  /*! \brief No inputs (ie fill to constant value) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out) {
    KERNEL_ASSIGN(out[i], req, OP::Map());
  }

  /*! \brief input is single scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(value));
  }

  /*! \brief inputs are two tensors and a scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out,
                                  const DType *input_1, const DType *input_2, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(input_1[i], input_2[i], value));
  }

  /*! \brief inputs are three tensors (ie backward grad with binary grad function) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out,
                                  const DType *input_1,
                                  const DType *input_2,
                                  const DType *input_3) {
    KERNEL_ASSIGN(out[i], req, OP::Map(input_1[i], input_2[i], input_3[i]));
  }

  /*! \brief input is a tensor and the output is a boolean tensor */
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const DType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i]));
  }

  /*! \brief inputs are two tensors with a boolean output tensor */
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is tensor and two scalar value with a boolean output tensor */
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }

#ifndef _WIN32
  /*! \brief inputs are two tensors with a half_t output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i,
                                  mshadow::half::half_t *out,
                                  const DType *lhs,
                                  const mshadow::half::half_t *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *lhs, const float *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a double output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, double *out, const DType *lhs, const double *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a half_t output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i,
                                  mshadow::half::half_t *out,
                                  const DType *lhs,
                                  const mshadow::half::half_t value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], value));
  }

  /*! \brief inputs are two tensors with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *lhs, const float value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], value));
  }

  /*! \brief inputs are two tensors with a double output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, double *out, const DType *lhs, const double value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], value));
  }
#endif

  /*! \brief inputs are two tensors with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is a tensor and a scalar value with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }
};
}  // namespace mxnet_op
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MXNET_OP_KERNEL_ASSIGN_H_

