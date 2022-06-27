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
 * \file log_softmax.cc
 * \brief CPU Implementation of log_softmax
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_op.h"
#include "../operator_common.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_softmax-inl.h"
#endif

namespace mxnet {
namespace op {

#if MXNET_USE_ONEDNN == 1
static void LogSoftmaxComputeExCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (SupportDNNLLogSoftmax(param, inputs[0])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLLogSoftmaxForward, attrs, ctx, inputs[0], req[0], outputs[0]);
    auto fn = SoftmaxCompute<cpu, mxnet_op::log_softmax_fwd>;
    DNNL_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(SoftmaxCompute<cpu, mxnet_op::log_softmax_fwd>, attrs, ctx, inputs, req, outputs);
}

static void LogSoftmaxGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (SupportDNNLLogSoftmax(param, inputs[1])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLLogSoftmaxBackward, attrs, ctx, inputs, req, outputs);
    auto fn = SoftmaxGradCompute<cpu, op::mshadow_op::left, mxnet_op::log_softmax_bwd>;
    DNNL_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(SoftmaxGradCompute<cpu, op::mshadow_op::left, mxnet_op::log_softmax_bwd>,
                  attrs,
                  ctx,
                  inputs,
                  req,
                  outputs);
}

inline static bool LogSoftmaxStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int>* in_attrs,
                                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

inline static bool LogSoftmaxGradStorageType(const nnvm::NodeAttrs& attrs,
                                             const int dev_mask,
                                             DispatchMode* dispatch_mode,
                                             std::vector<int>* in_attrs,
                                             std::vector<int>* out_attrs) {
  bool support   = true;
  int num_inputs = 2U;
  if (softmax_has_dtype_override(attrs)) {
    support    = false;
    num_inputs = 3U;
  }

  CHECK_EQ(in_attrs->size(), num_inputs);
  CHECK_EQ(out_attrs->size(), 1U);
  return DNNLStorageType(attrs, dev_mask, support, dispatch_mode, in_attrs, out_attrs);
}
#endif

NNVM_REGISTER_OP(log_softmax)
    .add_alias("_npx_log_softmax")
    .describe(R"code(Computes the log softmax of the input.
This is equivalent to computing softmax followed by log.

Examples::

  >>> x = mx.nd.array([1, 2, .1])
  >>> mx.nd.log_softmax(x).asnumpy()
  array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)

  >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
  >>> mx.nd.log_softmax(x, axis=0).asnumpy()
  array([[-0.34115392, -0.69314718, -1.24115396],
         [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)


)code")
    .set_attr_parser(ParamParser<SoftmaxParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::log_softmax_fwd>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", LogSoftmaxComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", LogSoftmaxStorageType)
#endif
    .set_attr<nnvm::FGradient>("FGradient", SoftmaxFGradient{"_backward_log_softmax"})
    .set_attr<nnvm::FInferType>("FInferType", SoftmaxOpType)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("data", "NDArray-or-Symbol", "The input array.")
    .add_arguments(SoftmaxParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_log_softmax)
    .set_num_inputs(SoftmaxGradOpNumInputs)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames", SoftmaxGradOpInputNames)
    .set_attr<mxnet::FInferShape>("FInferShape", SoftmaxGradOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SoftmaxGradOpType)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption", SoftmaxGradOpInplaceOption)
    .add_argument("args", "NDArray-or-Symbol[]", "Positional input arguments")
    .set_attr_parser(ParamParser<SoftmaxParam>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", LogSoftmaxGradComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", LogSoftmaxGradStorageType)
#endif
    .set_attr<FCompute>("FCompute<cpu>",
                        SoftmaxGradCompute<cpu, mshadow_op::left, mxnet_op::log_softmax_bwd>);

NNVM_REGISTER_OP(masked_log_softmax)
    .add_alias("_npx_masked_log_softmax")
    .describe(R"code(Computes the masked log softmax of the input.
This is equivalent to computing masked softmax followed by log.)code")
    .set_attr_parser(ParamParser<MaskedSoftmaxParam>)
    .set_attr<nnvm::FListOutputNames>("FListInputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"data", "mask"};
                                      })
    .set_attr<FCompute>("FCompute<cpu>", MaskedSoftmaxCompute<cpu, mxnet_op::log_softmax_fwd, true>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          auto data_grad = MakeNode("_backward_masked_log_softmax",
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

NNVM_REGISTER_OP(_backward_masked_log_softmax)
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
                        MaskedSoftmaxGradCompute<cpu, mshadow_op::left, mxnet_op::log_softmax_bwd>);

}  // namespace op
}  // namespace mxnet
