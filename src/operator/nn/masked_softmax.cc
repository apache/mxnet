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
 * \file masked_softmax.cc
 */

#include "softmax-inl.h"
#include "operator/tensor/elemwise_unary_op.h"
#include "operator/tensor/elemwise_binary_op.h"
#include "operator/operator_common.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_softmax-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(MaskedSoftmaxParam);

#if MXNET_USE_ONEDNN == 1
static void MaskedSoftmaxComputeExCPU(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<NDArray>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<NDArray>& outputs) {
  if (inputs[0].shape().Size() == 0U)
    return;
  const MaskedSoftmaxParam& param = nnvm::get<MaskedSoftmaxParam>(attrs.parsed);
  if (SupportDNNLMaskedSoftmax(param, inputs)) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);

    DNNLRun(DNNLMaskedSoftmaxForward, attrs, ctx, inputs, req, outputs);

    auto fn = MaskedSoftmaxCompute<cpu, mxnet_op::softmax_fwd, false>;
    DNNL_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(
      MaskedSoftmaxCompute<cpu, mxnet_op::softmax_fwd, false>, attrs, ctx, inputs, req, outputs);
}

inline static bool MaskedSoftmaxStorageType(const nnvm::NodeAttrs& attrs,
                                            const int dev_mask,
                                            DispatchMode* dispatch_mode,
                                            std::vector<int>* in_attrs,
                                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}
#endif

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
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", MaskedSoftmaxComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", MaskedSoftmaxStorageType)
#endif
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
