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
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_unary_op.cu
 * \brief Function definition of unary operators
 */

#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(__name$, __kernel$)     \
NNVM_REGISTER_OP(__name$)                                               \
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, __kernel$>)  \

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_negative, mshadow_op::negation);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_reciprocal, mshadow_op::reciprocal);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_absolute, mshadow_op::abs);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_sign, mshadow_op::sign);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_rint, mshadow_op::rint);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_ceil, mshadow_op::ceil);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_floor, mshadow_op::floor);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_trunc, mshadow_op::trunc);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_fix, mshadow_op::fix);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_square, mshadow_op::square);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_sqrt, mshadow_op::square_root);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_cbrt, mshadow_op::cube_root);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_exp, mshadow_op::exp);

NNVM_REGISTER_OP(_numpy_log)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::LogCompute<gpu, mshadow_op::log>);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_log10, mshadow_op::log10);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_log2, mshadow_op::log2);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_log1p, mshadow_op::log1p);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_expm1, mshadow_op::expm1);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_logical_not, mshadow_op::nt);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_sin, mshadow_op::sin);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_cos, mshadow_op::cos);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_tan, mshadow_op::tan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_arcsin, mshadow_op::arcsin);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_arccos, mshadow_op::arccos);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_arctan, mshadow_op::arctan);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_degrees, mshadow_op::degrees);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_radians, mshadow_op::radians);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_sinh, mshadow_op::sinh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_cosh, mshadow_op::cosh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_tanh, mshadow_op::tanh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_arcsinh, mshadow_op::arcsinh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_arccosh, mshadow_op::arccosh);

MXNET_OPERATOR_REGISTER_NUMPY_UNARY_GPU(_numpy_arctanh, mshadow_op::arctanh);

}
}
