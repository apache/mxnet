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
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpForward<gpu>);

NNVM_REGISTER_OP(_backward_slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpBackward<gpu>);

NNVM_REGISTER_OP(_slice_assign)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignOpForward<gpu>);

NNVM_REGISTER_OP(_slice_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignScalarOpForward<gpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxis<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxisGrad_<gpu>);

NNVM_REGISTER_OP(clip)
.set_attr<FCompute>("FCompute<gpu>", Clip<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ClipEx<gpu>);

NNVM_REGISTER_OP(_backward_clip)
.set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);

NNVM_REGISTER_OP(reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(_backward_reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_backward_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpBackward<gpu>);
}  // namespace op
}  // namespace mxnet
