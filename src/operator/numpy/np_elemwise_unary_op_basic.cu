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
 * \file np_elemwise_unary_op_basic.cu
 * \brief GPU Implementation of numpy unary functions.
 */
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npx_relu)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::relu>);

NNVM_REGISTER_OP(_npx_sigmoid)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::sigmoid>);

NNVM_REGISTER_OP(_np_copy)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(__name$, __kernel$)       \
  NNVM_REGISTER_OP(__name$)                                               \
  .set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, __kernel$>)

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_negative, mshadow_op::negation);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_reciprocal, mshadow_op::reciprocal);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_absolute, mshadow_op::abs);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sign, mshadow_op::sign);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_rint, mshadow_op::rint);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_ceil, mshadow_op::ceil);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_floor, mshadow_op::floor);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_trunc, mshadow_op::trunc);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_fix, mshadow_op::fix);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_square, mshadow_op::square);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sqrt, mshadow_op::square_root);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_cbrt, mshadow_op::cube_root);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_exp, mshadow_op::exp);

NNVM_REGISTER_OP(_npi_log)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log>);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log10, mshadow_op::log10);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log2, mshadow_op::log2);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_log1p, mshadow_op::log1p);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_expm1, mshadow_op::expm1);

NNVM_REGISTER_OP(_npi_logical_not)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::ComputeLogic<gpu, mshadow_op::np_logical_not>);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sin, mshadow_op::sin);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_cos, mshadow_op::cos);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_tan, mshadow_op::tan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arcsin, mshadow_op::arcsin);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arccos, mshadow_op::arccos);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arctan, mshadow_op::arctan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_degrees, mshadow_op::degrees);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_radians, mshadow_op::radians);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_sinh, mshadow_op::sinh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_cosh, mshadow_op::cosh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_tanh, mshadow_op::tanh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arcsinh, mshadow_op::arcsinh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arccosh, mshadow_op::arccosh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_npi_arctanh, mshadow_op::arctanh);

NNVM_REGISTER_OP(_npi_around)
.set_attr<FCompute>("FCompute<gpu>", AroundOpForward<gpu>);

}  // namespace op
}  // namespace mxnet
