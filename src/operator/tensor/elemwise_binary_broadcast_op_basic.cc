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
 * \file elemwise_binary_broadcast_op_basic.cc
 * \brief CPU Implementation of basic functions for elementwise binary broadcast operator.
 */
#include "operator/tensor/elemwise_unary_op.h"
#include "operator/tensor/elemwise_binary_op-inl.h"
#include "operator/tensor/elemwise_binary_broadcast_op.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_binary-inl.h"
#include "operator/nn/dnnl/dnnl_sum-inl.h"
#endif  // MXNET_USE_ONEDNN == 1

namespace mxnet {
namespace op {

#if MXNET_USE_ONEDNN == 1
template <dnnl::algorithm alg>
void DNNLBinaryOpForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  const mxnet::TShape& input_0_shape  = inputs[0].shape();
  const mxnet::TShape& input_1_shape  = inputs[1].shape();
  const mxnet::TShape& output_0_shape = outputs[0].shape();
  // We can use more efficient sum kernel, when there is no broadcast - when shapes are the
  // same.
  const bool same_shape = (input_0_shape == input_1_shape);

  if (same_shape && alg == dnnl::algorithm::binary_add) {
    DNNLSumFwd& fwd = DNNLSumFwd::GetCached(inputs, outputs);
    fwd.Execute(ctx, inputs, req, outputs);
  } else {
    mxnet::TShape new_lshape, new_rshape, new_oshape;
    int ndim_diff = BinaryBroadcastShapeCompact(
        input_0_shape, input_1_shape, output_0_shape, &new_lshape, &new_rshape, &new_oshape);
    std::vector<NDArray> new_inputs;
    std::vector<NDArray> new_outputs;
    if (ndim_diff) {
      new_inputs  = {inputs[0].Reshape(new_lshape), inputs[1].Reshape(new_rshape)};
      new_outputs = {outputs[0].Reshape(new_oshape)};
    } else if (input_0_shape.Size() == 1 && input_1_shape.Size() == 1) {
      // BinaryBroadcastShapeCompact function doesn't reshape tensors of size (1,1,...,1)
      // into shape (1). It is mandatory for oneDNN primitive to have this reshape done.
      mxnet::TShape one_shape = mxnet::TShape(1, 1);
      new_inputs              = {inputs[0].Reshape(one_shape), inputs[1].Reshape(one_shape)};
      new_outputs             = {outputs[0].Reshape(one_shape)};
    } else {
      new_inputs  = {inputs[0], inputs[1]};
      new_outputs = {outputs[0]};
    }

    DNNLBinaryOpFwd& fwd = DNNLBinaryOpFwd::GetBinaryOpForward<alg>(new_inputs, new_outputs);
    fwd.Execute(new_inputs, req, new_outputs);
  }
}
#endif

template <typename OP>
static void BinaryOperatorComputeExCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
#if MXNET_USE_ONEDNN == 1
  if (common::ContainsOnlyStorage(inputs, kDefaultStorage)) {
    if (SupportDNNLBinary(inputs, outputs)) {
      const dnnl::algorithm alg = DNNLAlgorithm<OP>::value;
      DNNLRun(DNNLBinaryOpForward<alg>, attrs, ctx, inputs, req, outputs);
    } else {
      FallBackCompute(BinaryBroadcastCompute<cpu, OP>, attrs, ctx, inputs, req, outputs);
    }
    return;
  }
#endif  // MXNET_USE_ONEDNN == 1
  if (std::is_same<OP, op::mshadow_op::plus>::value ||
      std::is_same<OP, op::mshadow_op::minus>::value) {
    BinaryBroadcastComputeDenseEx<cpu, OP>(attrs, ctx, inputs, req, outputs);
  } else if (std::is_same<OP, op::mshadow_op::mul>::value ||
             std::is_same<OP, op::mshadow_op::div>::value) {
    BinaryBroadcastComputeSparseEx<cpu, OP>(attrs, ctx, inputs, req, outputs);
  }
}

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_add)
MXNET_ADD_SPARSE_OP_ALIAS(broadcast_add)
MXNET_ADD_SPARSE_OP_ALIAS(broadcast_plus)
    .add_alias("broadcast_plus")
    .describe(R"code(Returns element-wise sum of the input arrays with broadcasting.

`broadcast_plus` is an alias to the function `broadcast_add`.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_add(x, y) = [[ 1.,  1.,  1.],
                          [ 2.,  2.,  2.]]

   broadcast_plus(x, y) = [[ 1.,  1.,  1.],
                           [ 2.,  2.,  2.]]

Supported sparse operations:

   broadcast_add(csr, dense(1D)) = dense
   broadcast_add(dense(1D), csr) = dense

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::plus>)
    .set_attr<FComputeEx>("FComputeEx<cpu>", BinaryOperatorComputeExCPU<op::mshadow_op::plus>)
    .set_attr<FInferStorageType>("FInferStorageType", BinaryBroadcastAddStorageType)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_add"});

NNVM_REGISTER_OP(_backward_broadcast_add)
    .set_num_inputs(1)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity, mshadow_op::identity>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_sub)
MXNET_ADD_SPARSE_OP_ALIAS(broadcast_sub)
MXNET_ADD_SPARSE_OP_ALIAS(broadcast_minus)
    .add_alias("broadcast_minus")
    .describe(R"code(Returns element-wise difference of the input arrays with broadcasting.

`broadcast_minus` is an alias to the function `broadcast_sub`.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                          [ 0.,  0.,  0.]]

   broadcast_minus(x, y) = [[ 1.,  1.,  1.],
                            [ 0.,  0.,  0.]]

Supported sparse operations:

   broadcast_sub/minus(csr, dense(1D)) = dense
   broadcast_sub/minus(dense(1D), csr) = dense

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::minus>)
    .set_attr<FComputeEx>("FComputeEx<cpu>", BinaryOperatorComputeExCPU<op::mshadow_op::minus>)
    .set_attr<FInferStorageType>("FInferStorageType", BinaryBroadcastAddStorageType)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_sub"});

NNVM_REGISTER_OP(_backward_broadcast_sub)
    .set_num_inputs(1)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity, mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_mul)
MXNET_ADD_SPARSE_OP_ALIAS(broadcast_mul)
    .describe(R"code(Returns element-wise product of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                          [ 1.,  1.,  1.]]

Supported sparse operations:

   broadcast_mul(csr, dense(1D)) = csr

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::mul>)
    .set_attr<FComputeEx>("FComputeEx<cpu>", BinaryOperatorComputeExCPU<op::mshadow_op::mul>)
    .set_attr<FInferStorageType>("FInferStorageType", BinaryBroadcastMulStorageType)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mul"});

NNVM_REGISTER_OP(_backward_broadcast_mul)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryBroadcastBackwardUseIn<cpu, mshadow_op::right, mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_div)
MXNET_ADD_SPARSE_OP_ALIAS(broadcast_div)
    .describe(R"code(Returns element-wise division of the input arrays with broadcasting.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_div(x, y) = [[ 3.,  3.,  3.],
                          [ 2.,  2.,  2.]]

Supported sparse operations:

   broadcast_div(csr, dense(1D)) = csr

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::div>)
    .set_attr<FComputeEx>("FComputeEx<cpu>", BinaryOperatorComputeExCPU<op::mshadow_op::div>)
    .set_attr<FInferStorageType>("FInferStorageType", BinaryBroadcastMulStorageType)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_div"});

NNVM_REGISTER_OP(_backward_broadcast_div)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        BinaryBroadcastBackwardUseIn<cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_mod)
    .describe(R"code(Returns element-wise modulo of the input arrays with broadcasting.

Example::

   x = [[ 8.,  8.,  8.],
        [ 8.,  8.,  8.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_mod(x, y) = [[ 0.,  0.,  0.],
                          [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

NNVM_REGISTER_OP(_backward_broadcast_mod)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        BinaryBroadcastBackwardUseIn<cpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
