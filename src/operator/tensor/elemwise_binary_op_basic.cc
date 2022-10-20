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
 * \file elemwise_binary_op_basic.cc
 * \brief CPU Implementation of basic elementwise binary broadcast operators
 */
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_copy-inl.h"
#include "operator/nn/dnnl/dnnl_sum-inl.h"
#include "./elemwise_binary_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {

#if MXNET_USE_ONEDNN == 1
// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_eltwise.html
bool SupportDNNLSum(const std::vector<NDArray>& inputs) {
  return SupportDNNL(inputs[0]) && SupportDNNL(inputs[1]);
}
#endif

static void ElemwiseAddEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
#if MXNET_USE_ONEDNN == 1
  if (SupportDNNLSum(inputs) && common::ContainsOnlyStorage(inputs, kDefaultStorage)) {
    DNNLRun(DNNLSumForward, attrs, ctx, inputs, req, outputs);
    return;
  } else if (inputs[0].storage_type() == kDefaultStorage &&
             inputs[1].storage_type() == kDefaultStorage) {
    FallBackCompute(
        ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::plus>, attrs, ctx, inputs, req, outputs);
    return;
  }
#endif
  ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(attrs, ctx, inputs, req, outputs);
}

static inline bool ElemwiseAddStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  bool ret = ElemwiseBinaryOp::PreferDenseStorageType<true, true, true>(
      attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
#if MXNET_USE_ONEDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && !DNNLEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  } else if (dev_mask == mshadow::cpu::kDevMask &&
             common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) &&
             out_attrs->at(0) == kDefaultStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

MXNET_OPERATOR_REGISTER_BINARY(elemwise_add)
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
    .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::plus>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
#endif
    .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseAddEx)
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<FResourceRequest>("FResourceRequest", /* For Sparse CSR */
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                }) MXNET_ADD_SPARSE_OP_ALIAS(elemwise_add)
    .add_alias("_add")
    .add_alias("_plus")
    .add_alias("_Plus")
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output"};
                                      })
    .describe(R"code(Adds arguments element-wise.

The storage type of ``elemwise_add`` output depends on storage types of inputs

   - elemwise_add(row_sparse, row_sparse) = row_sparse
   - elemwise_add(csr, csr) = csr
   - elemwise_add(default, csr) = default
   - elemwise_add(csr, default) = default
   - elemwise_add(default, rsp) = default
   - elemwise_add(rsp, default) = default
   - otherwise, ``elemwise_add`` generates output with default storage

)code")
    .set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_add"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_grad_add, op::mshadow_op::plus);

static void _backward_ElemwiseAddEx(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
#if MXNET_USE_ONEDNN == 1
  if (inputs[0].IsDNNLData()) {
    DNNLRun(DNNLCopy, attrs, ctx, inputs[0], req[0], outputs[0]);
    DNNLRun(DNNLCopy, attrs, ctx, inputs[0], req[1], outputs[1]);
    return;
  } else if (common::ContainsOnlyStorage(inputs, kDefaultStorage)) {
    FallBackCompute(
        ElemwiseBinaryOp::BackwardUseNone<cpu, mshadow_op::identity, mshadow_op::identity>,
        attrs,
        ctx,
        inputs,
        req,
        outputs);
    return;
  }
#endif
  ElemwiseBinaryOp::BackwardUseNoneEx<cpu, mshadow_op::identity, mshadow_op::identity>(
      attrs, ctx, inputs, req, outputs);
}

static inline bool ElemwiseAddBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                                  const int dev_mask,
                                                  DispatchMode* dispatch_mode,
                                                  std::vector<int>* in_attrs,
                                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  bool ret = ElemwiseStorageType<1, 2, true, true, true>(
      attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
#if MXNET_USE_ONEDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && !DNNLEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  } else if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

NNVM_REGISTER_OP(_backward_add)
    .set_num_inputs(1)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
                                    })
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true)
#endif
    .set_attr<FCompute>(
        "FCompute<cpu>",
        ElemwiseBinaryOp::BackwardUseNone<cpu, mshadow_op::identity, mshadow_op::identity>)
    .set_attr<FComputeEx>("FComputeEx<cpu>", _backward_ElemwiseAddEx)
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddBackwardStorageType);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_PD(elemwise_sub, op::mshadow_op::minus)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_sub)
    .add_alias("_sub")
    .add_alias("_minus")
    .add_alias("_Minus")
    .describe(R"code(Subtracts arguments element-wise.

The storage type of ``elemwise_sub`` output depends on storage types of inputs

   - elemwise_sub(row_sparse, row_sparse) = row_sparse
   - elemwise_sub(csr, csr) = csr
   - elemwise_sub(default, csr) = default
   - elemwise_sub(csr, default) = default
   - elemwise_sub(default, rsp) = default
   - elemwise_sub(rsp, default) = default
   - otherwise, ``elemwise_sub`` generates output with default storage

)code")
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
    .set_num_inputs(1)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
                                    })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        ElemwiseBinaryOp::BackwardUseNone<cpu, mshadow_op::identity, mshadow_op::negation>)
    .set_attr<FComputeEx>(
        "FComputeEx<cpu>",
        ElemwiseBinaryOp::BackwardUseNoneEx<cpu, mshadow_op::identity, mshadow_op::negation>)
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 2, true, true, true>);

MXNET_OPERATOR_REGISTER_BINARY(elemwise_mul)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_mul)
    .describe(R"code(Multiplies arguments element-wise.

The storage type of ``elemwise_mul`` output depends on storage types of inputs

   - elemwise_mul(default, default) = default
   - elemwise_mul(row_sparse, row_sparse) = row_sparse
   - elemwise_mul(default, row_sparse) = row_sparse
   - elemwise_mul(row_sparse, default) = row_sparse
   - elemwise_mul(csr, csr) = csr
   - otherwise, ``elemwise_mul`` generates output with default storage

)code")
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseBinaryOp::PreferSparseStorageType)
    .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>)
    .set_attr<FComputeEx>(
        "FComputeEx<cpu>",
        ElemwiseBinaryOp::ComputeDnsLRValueEx<cpu, op::mshadow_op::mul, true, true>)
    .set_attr<FResourceRequest>("FResourceRequest", /* For Sparse CSR */
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .add_alias("_mul")
    .add_alias("_Mul")
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseBinaryOp::BackwardUseInStorageType)
    .set_attr<FResourceRequest>("FResourceRequest", /* For Sparse CSR */
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>",
                        ElemwiseBinaryOp::BackwardUseIn<cpu, mshadow_op::right, mshadow_op::left>)
    .set_attr<FComputeEx>(
        "FComputeEx<cpu>",
        ElemwiseBinaryOp::BackwardUseInEx<cpu, mshadow_op::right, mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(elemwise_div, op::mshadow_op::div)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_div)
    .describe(R"code(Divides arguments element-wise.

The storage type of ``elemwise_div`` output is always dense

)code")
    .add_alias("_div")
    .add_alias("_Div")
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        ElemwiseBinaryOp::BackwardUseIn<cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_mod)
    .add_alias("_Mod")
    .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, mshadow_op::mod>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod"});

NNVM_REGISTER_OP(_backward_mod)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FCompute>(
        "FCompute<cpu>",
        ElemwiseBinaryOp::BackwardUseIn<cpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
