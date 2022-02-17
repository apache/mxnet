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
 * \file elemwise_unary_op_basic.cu
 * \brief GPU Implementation of unary functions.
 */
#include "./elemwise_binary_op.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(relu)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"relu"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"relu"});

NNVM_REGISTER_OP(_backward_relu)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_relu"});

NNVM_REGISTER_OP(sigmoid).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sigmoid"});

NNVM_REGISTER_OP(_backward_sigmoid)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_sigmoid"});

NNVM_REGISTER_OP(log_sigmoid).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"log_sigmoid"});

NNVM_REGISTER_OP(_backward_log_sigmoid)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_log_sigmoid"});

NNVM_REGISTER_OP(mish).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"mish"});

NNVM_REGISTER_OP(_backward_mish)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_mish"});

NNVM_REGISTER_OP(hard_sigmoid).set_attr<FCompute>("FCompute<gpu>", HardSigmoidForward<gpu>);

NNVM_REGISTER_OP(_backward_hard_sigmoid)
    .set_attr<FCompute>("FCompute<gpu>", HardSigmoidBackward<gpu>);

// softsign
NNVM_REGISTER_OP(softsign).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"softsign"});

NNVM_REGISTER_OP(_backward_softsign)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_softsign"});

// erf
NNVM_REGISTER_OP(erf).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"erf"});

NNVM_REGISTER_OP(_backward_erf)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_erf"});

// erfinv
NNVM_REGISTER_OP(erfinv).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"erfinv"});

NNVM_REGISTER_OP(_backward_erfinv)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_erfinv"});

// copy
NNVM_REGISTER_OP(_copy)
    .set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeEx<gpu>);

NNVM_REGISTER_OP(_backward_copy)
    .set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeEx<gpu>);

NNVM_REGISTER_OP(_backward_reshape)
    .set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(BlockGrad).set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(make_loss)
    .set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeEx<gpu>);

// identity output as first input, but attributes are constrainted to be like rhs
NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
    .set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeFirstItemEx<gpu>);

NNVM_REGISTER_OP(reshape_like).set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

void ShapeComputeGPU(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data    = inputs[0];
  const TBlob& out_data   = outputs[0];
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  cudaMemcpyAsync(out_data.dptr_,
                  in_data.shape_.data(),
                  in_data.ndim() * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  mshadow::Stream<gpu>::GetStream(s));
}

NNVM_REGISTER_OP(shape_array)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
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
  const TBlob& in_data    = inputs[0];
  const TBlob& out_data   = outputs[0];
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  const index_t size_var  = in_data.Size();
  cudaMemcpyAsync(out_data.dptr_,
                  &size_var,
                  1U * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  mshadow::Stream<gpu>::GetStream(s));
}

NNVM_REGISTER_OP(size_array).set_attr<FCompute>("FCompute<gpu>", SizeComputeGPU);

NNVM_REGISTER_OP(Cast).set_attr<FCompute>("FCompute<gpu>", CastCompute<gpu>);

NNVM_REGISTER_OP(_backward_cast).set_attr<FCompute>("FCompute<gpu>", CastCompute<gpu>);

// negative
NNVM_REGISTER_OP(negative)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"negation"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"negation"});

// abs
NNVM_REGISTER_OP(abs)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"abs"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"abs"});

NNVM_REGISTER_OP(_backward_abs)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_abs"});

// sign
NNVM_REGISTER_OP(sign)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sign"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"sign"});

// round
NNVM_REGISTER_OP(round)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"round"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"round"});

// ceil
NNVM_REGISTER_OP(ceil)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"ceil"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"ceil"});

// floor
NNVM_REGISTER_OP(floor)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"floor"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"floor"});

// trunc
NNVM_REGISTER_OP(trunc)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"trunc"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"trunc"});

// rint
NNVM_REGISTER_OP(rint)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"rint"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"rint"});

// fix
NNVM_REGISTER_OP(fix)
    .set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"fix"})
    .set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"fix"});

// gamma
NNVM_REGISTER_OP(gamma).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"gamma"});

NNVM_REGISTER_OP(_backward_gamma)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_gamma"});

// gammaln
NNVM_REGISTER_OP(gammaln).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"gammaln"});

NNVM_REGISTER_OP(_backward_gammaln)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_gammaln"});

// digamma
NNVM_REGISTER_OP(digamma).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"digamma"});

NNVM_REGISTER_OP(_backward_digamma)
    .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_digamma"});

// logical not
NNVM_REGISTER_OP(logical_not).set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"logical_not"});

}  // namespace op
}  // namespace mxnet
