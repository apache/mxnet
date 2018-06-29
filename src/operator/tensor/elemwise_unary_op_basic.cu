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
 * \brief GPU Implementation of unary trigometric functions.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(relu)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::relu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::relu>);

NNVM_REGISTER_OP(_backward_relu)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::relu_grad>>);

NNVM_REGISTER_OP(sigmoid)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::sigmoid>);

NNVM_REGISTER_OP(_backward_sigmoid)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::sigmoid_grad>>);

NNVM_REGISTER_OP(hard_sigmoid)
.set_attr<FCompute>("FCompute<gpu>", HardSigmoidForward<gpu>);

NNVM_REGISTER_OP(_backward_hard_sigmoid)
.set_attr<FCompute>("FCompute<gpu>", HardSigmoidBackward<gpu>);

// softsign
NNVM_REGISTER_OP(softsign)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::softsign>);

NNVM_REGISTER_OP(_backward_softsign)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::softsign_grad>>);

// copy
NNVM_REGISTER_OP(_copy)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeEx<gpu>);

NNVM_REGISTER_OP(_backward_copy)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(BlockGrad)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(make_loss)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeEx<gpu>);

// identity output as first input, but attributes are constrainted to be like rhs
NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeFirstItemEx<gpu>);

NNVM_REGISTER_OP(reshape_like)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

void ShapeComputeGPU(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaMemcpyAsync(out_data.dptr_,
                  in_data.shape_.data(),
                  in_data.ndim() * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  mshadow::Stream<gpu>::GetStream(s));
}

NNVM_REGISTER_OP(shape_array)
.set_attr<FCompute>("FCompute<gpu>", ShapeComputeGPU);

void SizeComputeGPU(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  const index_t size_var = in_data.Size();
  cudaMemcpyAsync(out_data.dptr_,
                  &size_var,
                  1U * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  mshadow::Stream<gpu>::GetStream(s));
}

NNVM_REGISTER_OP(size_array)
.set_attr<FCompute>("FCompute<gpu>", SizeComputeGPU);

NNVM_REGISTER_OP(Cast)
.set_attr<FCompute>("FCompute<gpu>", CastCompute<gpu>);

NNVM_REGISTER_OP(_backward_cast)
.set_attr<FCompute>("FCompute<gpu>", CastCompute<gpu>);

// negative
NNVM_REGISTER_OP(negative)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::negation>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::negation>);

// reciprocal
NNVM_REGISTER_OP(reciprocal)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::reciprocal>);

NNVM_REGISTER_OP(_backward_reciprocal)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::reciprocal_grad> >);

// abs
NNVM_REGISTER_OP(abs)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::abs>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::abs>);

NNVM_REGISTER_OP(_backward_abs)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<gpu, unary_bwd<mshadow_op::sign> >);

// sign
NNVM_REGISTER_OP(sign)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::sign>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::sign>);

NNVM_REGISTER_OP(_backward_sign)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::sign_grad> >);

// round
NNVM_REGISTER_OP(round)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::round>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::round>);

// ceil
NNVM_REGISTER_OP(ceil)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::ceil>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::ceil>);

// floor
NNVM_REGISTER_OP(floor)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::floor>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::floor>);

// trunc
NNVM_REGISTER_OP(trunc)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::trunc>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::trunc>);

// rint
NNVM_REGISTER_OP(rint)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::rint>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::rint>);


// fix
NNVM_REGISTER_OP(fix)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::fix>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::fix>);


// square
NNVM_REGISTER_OP(square)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::square>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::square>);

NNVM_REGISTER_OP(_backward_square)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::square_grad> >);

// sqrt
NNVM_REGISTER_OP(sqrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::square_root>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::square_root>);


NNVM_REGISTER_OP(_backward_sqrt)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::square_root_grad> >);

// rsqrt
NNVM_REGISTER_OP(rsqrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::reciprocal_square_root>);

NNVM_REGISTER_OP(_backward_rsqrt)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryOp::Compute<gpu, unary_bwd<mshadow_op::reciprocal_square_root_grad> >);

// cbrt
NNVM_REGISTER_OP(cbrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::cube_root>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::cube_root>);


NNVM_REGISTER_OP(_backward_cbrt)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::cube_root_grad> >);

// rcbrt
NNVM_REGISTER_OP(rcbrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::reciprocal_cube_root>);

NNVM_REGISTER_OP(_backward_rcbrt)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryOp::Compute<gpu, unary_bwd<mshadow_op::reciprocal_cube_root_grad> >);

// exp
NNVM_REGISTER_OP(exp)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::exp>);

// log
NNVM_REGISTER_OP(log)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log>);

// log10
NNVM_REGISTER_OP(log10)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log10>);

// log2
NNVM_REGISTER_OP(log2)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log2>);

NNVM_REGISTER_OP(_backward_log)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log_grad> >);

NNVM_REGISTER_OP(_backward_log10)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log10_grad> >);

NNVM_REGISTER_OP(_backward_log2)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log2_grad> >);

// log1p
NNVM_REGISTER_OP(log1p)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log1p>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::log1p>);

NNVM_REGISTER_OP(_backward_log1p)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log1p_grad> >);

// expm1
NNVM_REGISTER_OP(expm1)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::expm1>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::expm1>);

NNVM_REGISTER_OP(_backward_expm1)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::exp> >);

// gamma
NNVM_REGISTER_OP(gamma)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::gamma>);

NNVM_REGISTER_OP(_backward_gamma)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::gamma_grad> >);

// gammaln
NNVM_REGISTER_OP(gammaln)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::gammaln>);

NNVM_REGISTER_OP(_backward_gammaln)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::gammaln_grad> >);

// logical not
NNVM_REGISTER_OP(logical_not)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::nt>);

}  // namespace op
}  // namespace mxnet
