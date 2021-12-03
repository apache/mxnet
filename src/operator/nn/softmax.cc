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
 * \file softmax.cc
 * \brief CPU Implementation of softmax
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_op.h"
#include "../operator_common.h"
#if MXNET_USE_ONEDNN == 1
#include "dnnl/dnnl_base-inl.h"
#include "dnnl/dnnl_ops-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(SoftmaxParam);
DMLC_REGISTER_PARAMETER(MaskedSoftmaxParam);

#if MXNET_USE_ONEDNN == 1
static void SoftmaxComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (SupportDNNLSoftmax(param, inputs[0], outputs[0])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLSoftmaxForward, attrs, ctx, inputs[0], req[0], outputs[0]);
    auto fn = SoftmaxCompute<cpu, mxnet_op::softmax_fwd>;
    DNNL_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(SoftmaxCompute<cpu, mxnet_op::softmax_fwd>, attrs, ctx, inputs, req, outputs);
}

static void SoftmaxGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (SupportDNNLSoftmax(param, inputs[1], outputs[0])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLSoftmaxBackward, attrs, ctx, inputs, req, outputs);
    auto fn = SoftmaxGradCompute<cpu, op::mshadow_op::mul, mxnet_op::softmax_bwd>;
    DNNL_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(SoftmaxGradCompute<cpu, op::mshadow_op::mul, mxnet_op::softmax_bwd>,
                  attrs,
                  ctx,
                  inputs,
                  req,
                  outputs);
}

inline static bool SoftmaxStorageType(const nnvm::NodeAttrs& attrs,
                                      const int dev_mask,
                                      DispatchMode* dispatch_mode,
                                      std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), (param.use_length.value()) ? 2U : 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  if (param.use_length.value()) {
    auto& out_stype = out_attrs->at(0);
    return storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode, DispatchMode::kFCompute);
  }

  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

inline static bool SoftmaxGradStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  bool support = true;
  if (softmax_use_length(attrs) || softmax_has_dtype_override(attrs)) {
    support = false;
  }

  CHECK_EQ(in_attrs->size(), SoftmaxGradOpNumInputs(attrs));
  CHECK_EQ(out_attrs->size(), softmax_use_length(attrs) ? 2U : 1U);
  return DNNLStorageType(attrs, dev_mask, support, dispatch_mode, in_attrs, out_attrs);
}
#endif

NNVM_REGISTER_OP(softmax)
    .add_alias("_npx_softmax")
    .describe(R"code(Applies the softmax function.

The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.

.. math::
   softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}

for :math:`j = 1, ..., K`

t is the temperature parameter in softmax function. By default, t equals 1.0

Example::

  x = [[ 1.  1.  1.]
       [ 1.  1.  1.]]

  softmax(x,axis=0) = [[ 0.5  0.5  0.5]
                       [ 0.5  0.5  0.5]]

  softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
                       [ 0.33333334,  0.33333334,  0.33333334]]

)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<SoftmaxParam>)
    .set_attr<nnvm::FListOutputNames>("FListInputNames",
                                      [](const NodeAttrs& attrs) {
                                        const SoftmaxParam& param =
                                            nnvm::get<SoftmaxParam>(attrs.parsed);
                                        return (param.use_length.value()) ?
                                                   std::vector<std::string>{"data", "length"} :
                                                   std::vector<std::string>{"data"};
                                      })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output"};
                                      })
    .set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::softmax_fwd>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", SoftmaxComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", SoftmaxStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif
    .set_attr<nnvm::FGradient>("FGradient", SoftmaxFGradient{"_backward_softmax"})
    // .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<nnvm::FInferType>("FInferType", SoftmaxOpType)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
      return (param.use_length.value()) ? 2 : 1;
    })
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", SoftmaxOpShape)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("data", "NDArray-or-Symbol", "The input array.")
    .add_argument("length", "NDArray-or-Symbol", "The length array.")
    .add_arguments(SoftmaxParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_softmax)
    .set_num_inputs(SoftmaxGradOpNumInputs)
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      return (softmax_use_length(attrs) ? 2 : 1);
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames", SoftmaxGradOpInputNames)
    .set_attr<mxnet::FInferShape>("FInferShape", SoftmaxGradOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SoftmaxGradOpType)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption", SoftmaxGradOpInplaceOption)
    .add_argument("args", "NDArray-or-Symbol[]", "Positional input arguments")
    .set_attr_parser(ParamParser<SoftmaxParam>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", SoftmaxGradComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", SoftmaxGradStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif
    .set_attr<FCompute>("FCompute<cpu>",
                        SoftmaxGradCompute<cpu, op::mshadow_op::mul, mxnet_op::softmax_bwd>);

NNVM_REGISTER_OP(masked_softmax)
    .add_alias("_npx_masked_softmax")
    .describe(
        R"code(Applies the softmax function masking elements according to the mask provided)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<MaskedSoftmaxParam>)
    .set_attr<nnvm::FListOutputNames>("FListInputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"data", "mask"};
                                      })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output"};
                                      })
    .set_attr<FCompute>("FCompute<cpu>", MaskedSoftmaxCompute<cpu, mxnet_op::softmax_fwd, false>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          auto data_grad = MakeNode("_backward_masked_softmax",
                                    n->attrs.name + "_backward_data",
                                    {ograds[0], n->inputs[1], nnvm::NodeEntry(n, 0, 0)},
                                    &n->attrs.dict,
                                    &n);
          auto mask_grad =
              MakeNode("zeros_like", n->attrs.name + "_backward_mask", {n->inputs[1]}, nullptr, &n);
          std::vector<nnvm::NodeEntry> ret;
          ret.emplace_back(data_grad);
          ret.emplace_back(mask_grad);
          return ret;
        })
    .set_attr<nnvm::FInferType>("FInferType", MaskedSoftmaxOpType)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", MaskedSoftmaxOpShape)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("data", "NDArray-or-Symbol", "The input array.")
    .add_argument("mask", "NDArray-or-Symbol", "Mask to apply.")
    .add_arguments(MaskedSoftmaxParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_masked_softmax)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr<nnvm::FListOutputNames>("FListInputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"ograd", "mask", "output"};
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", MaskedSoftmaxGradOpShape)
    .set_attr<nnvm::FInferType>("FInferType", MaskedSoftmaxGradOpType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FInplaceOption>("FInplaceOption", MaskedSoftmaxGradOpInplaceOption)
    .add_argument("args", "NDArray-or-Symbol[]", "Positional input arguments")
    .set_attr_parser(ParamParser<MaskedSoftmaxParam>)
    .set_attr<FCompute>("FCompute<cpu>",
                        MaskedSoftmaxGradCompute<cpu, op::mshadow_op::mul, mxnet_op::softmax_bwd>);
}  // namespace op
}  // namespace mxnet
