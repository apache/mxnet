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
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu, Da Zheng
*/
#include "./activation-inl.h"
#include "../mshadow_op.h"
#include "../tensor/elemwise_unary_op.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_base-inl.h"
#include "./mkldnn/mkldnn_act-inl.h"
#endif  // MXNET_USE_MKLDNN

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ActivationParam);

// This will determine the order of the inputs for backward computation.
struct ActivationGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.emplace_back(nnvm::NodeEntry{n, activation::kOut, 0});
#if MXNET_USE_CUDNN == 1
    heads.push_back(n->inputs[activation::kData]);
#endif
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

static void ActivationComputeEx_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNN(inputs[0])) {
    MKLDNNAct_Forward<float>(ctx, param, inputs[0], req[0], outputs[0]);
    return;
  }
#endif
  _ActivationCompute<cpu>(param, ctx, inputs[0].data(), req[0], outputs[0].data());
}

void ActivationGradComputeEx_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
#if MXNET_USE_CUDNN == 1
  CHECK_EQ(inputs.size(), 3U);
#else
  CHECK_EQ(inputs.size(), 2U);
#endif
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNN(inputs[0])) {
    MKLDNNAct_Backward<float>(ctx, param, inputs[0], inputs[1], req[0],
        outputs[0]);
    return;
  }
#endif
  _ActivationGradCompute<cpu>(param, ctx, inputs[0].data(), inputs[1].data(),
      req[0], outputs[0].data());
}

inline static bool ActivationStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && SupportMKLDNNAct(param)
      // There is no reason to use MKLDNN activation if the input isn't in
      // MKLDNN format.
      && in_attrs->at(0) == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    (*out_attrs)[0] = kMKLDNNStorage;
    return true;
  }
#endif
  return ElemwiseStorageType<1, 1, false, false, false>(attrs, dev_mask,
      dispatch_mode, in_attrs, out_attrs);
}

inline static bool backward_ActStorageType(const nnvm::NodeAttrs& attrs,
                                           const int dev_mask,
                                           DispatchMode* dispatch_mode,
                                           std::vector<int> *in_attrs,
                                           std::vector<int> *out_attrs) {
#if MXNET_USE_CUDNN == 1
  CHECK_EQ(in_attrs->size(), 3U);
#else
  CHECK_EQ(in_attrs->size(), 2U);
#endif
  CHECK_EQ(out_attrs->size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && SupportMKLDNNAct(param)
      // There is no reason to use MKLDNN activation if the input isn't in
      // MKLDNN format.
      && in_attrs->at(0) == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    (*out_attrs)[0] = kMKLDNNStorage;
    return true;
  }
#endif
#if MXNET_USE_CUDNN == 1
  return ElemwiseStorageType<3, 1, false, false, false>(attrs, dev_mask,
      dispatch_mode, in_attrs, out_attrs);
#else
  return ElemwiseStorageType<2, 1, false, false, false>(attrs, dev_mask,
      dispatch_mode, in_attrs, out_attrs);
#endif
}

MXNET_OPERATOR_REGISTER_UNARY(Activation)
.describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FInferStorageType>("FInferStorageType", ActivationStorageType)
.set_attr<FCompute>("FCompute<cpu>", ActivationCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ActivationComputeEx_CPU)
.set_attr<nnvm::FGradient>("FGradient", ActivationGrad{"_backward_Activation"})
.add_arguments(ActivationParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Activation)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", backward_ActStorageType)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", ActivationGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ActivationGradComputeEx_CPU);

}  // namespace op
}  // namespace mxnet
