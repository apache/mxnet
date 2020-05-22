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
 * \file elemwise_unary_op_trig.cu
 * \brief GPU Implementation of unary trigonometric function.
 */
#include "./elemwise_binary_op.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {

// sin
NNVM_REGISTER_OP(sin)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sin"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"sin"});

NNVM_REGISTER_OP(_backward_sin)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::sin_grad> >);

// cos
NNVM_REGISTER_OP(cos)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"cos"});

NNVM_REGISTER_OP(_backward_cos)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::cos_grad> >);

// tan
NNVM_REGISTER_OP(tan)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"tan"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"tan"});

NNVM_REGISTER_OP(_backward_tan)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::tan_grad> >);

// arcsin
NNVM_REGISTER_OP(arcsin)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"arcsin"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"arcsin"});

NNVM_REGISTER_OP(_backward_arcsin)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::arcsin_grad> >);

// arccos
NNVM_REGISTER_OP(arccos)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"arccos"});

NNVM_REGISTER_OP(_backward_arccos)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::arccos_grad> >);

// arctan
NNVM_REGISTER_OP(arctan)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"arctan"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"arctan"});

NNVM_REGISTER_OP(_backward_arctan)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::arctan_grad> >);

// degrees
NNVM_REGISTER_OP(degrees)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"degrees"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"degrees"});

NNVM_REGISTER_OP(_backward_degrees)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::degrees_grad> >);

// radians
NNVM_REGISTER_OP(radians)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"radians"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"radians"});

NNVM_REGISTER_OP(_backward_radians)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::radians_grad> >);

// cosh
NNVM_REGISTER_OP(cosh)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"cosh"});

NNVM_REGISTER_OP(_backward_cosh)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::cosh_grad> >);

// sinh
NNVM_REGISTER_OP(sinh)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sinh"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"sinh"});

NNVM_REGISTER_OP(_backward_sinh)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::sinh_grad> >);

// tanh
NNVM_REGISTER_OP(tanh)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"tanh"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"tanh"});

NNVM_REGISTER_OP(_backward_tanh)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::tanh_grad> >);

// arcsinh
NNVM_REGISTER_OP(arcsinh)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"arcsinh"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"arcsinh"});

NNVM_REGISTER_OP(_backward_arcsinh)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::arcsinh_grad> >);

// arccosh
NNVM_REGISTER_OP(arccosh)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"arccosh"});

NNVM_REGISTER_OP(_backward_arccosh)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::arccosh_grad> >);

// arctanh
NNVM_REGISTER_OP(arctanh)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"arctanh"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"arctanh"});

NNVM_REGISTER_OP(_backward_arctanh)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::arctanh_grad> >);

}  // namespace op
}  // namespace mxnet
