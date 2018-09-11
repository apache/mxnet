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
#include "./mkldnn/mkldnn_ops-inl.h"
#endif  // MXNET_USE_MKLDNN
#include "../operator_common.h"
#include "../../common/utils.h"

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

    const NodeAttrs& attrs = n->attrs;
    int act_type = dmlc::get<ActivationParam>(attrs.parsed).act_type;
    if (act_type == activation::kSoftSign) {
      // for softsign need the inputs to compute the activation.
      heads.push_back(n->inputs[activation::kData]);
    }

#if (MXNET_USE_CUDNN == 1 || MXNET_USE_MKLDNN == 1)
    // for ReLU, no need to pass input data. This enables inplace optimization during the
    // forward pass.
    if (act_type != activation::kReLU &&
        act_type != activation::kSoftSign) {
      heads.push_back(n->inputs[activation::kData]);
    }
#endif
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

#if MXNET_USE_MKLDNN == 1
static void ActivationComputeExCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (SupportMKLDNN(inputs[0])) {
    MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    MKLDNNActivationForward(attrs, ctx, inputs[0], req[0], outputs[0]);
    MKLDNN_OPCHECK_RUN(ActivationCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(ActivationComputeImpl<cpu>, attrs, ctx, inputs, req, outputs);
}

void ActivationGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  bool relu = param.act_type == activation::kReLU;
  CHECK_EQ(inputs.size(), relu ? 2U : 3U);
  if (SupportMKLDNN(inputs[0])) {
    MKLDNN_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    // XXX: for y = relu(x), y is passed as "in_data" to Backward()
    MKLDNNActivationBackward(attrs, ctx, inputs[0], relu ? inputs[1] : inputs[2], req[0],
                             outputs[0]);
     MKLDNN_OPCHECK_RUN(ActivationGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(ActivationGradComputeImpl<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif

#if MXNET_USE_MKLDNN == 1
inline static bool ActivationStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  return MKLDNNStorageType(attrs, dev_mask, SupportMKLDNNAct(param),
                           dispatch_mode, in_attrs, out_attrs);
}

inline static bool BackwardActStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  if (param.act_type != activation::kReLU) {
    CHECK_EQ(in_attrs->size(), 3U);
  } else {
    // for ReLU activation, the backward pass only needs ograd and output
    CHECK_EQ(in_attrs->size(), 2U);
  }
  return MKLDNNStorageType(attrs, dev_mask, SupportMKLDNNAct(param),
                           dispatch_mode, in_attrs, out_attrs);
}
#endif

MXNET_OPERATOR_REGISTER_UNARY(Activation)
.describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
- `softsign`: :math:`y = \frac{x}{1 + abs(x)}`

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ActivationParam>)
#if MXNET_USE_MKLDNN == 1
.set_attr<FInferStorageType>("FInferStorageType", ActivationStorageType)
#endif
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<FCompute>("FCompute<cpu>", ActivationCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", ActivationComputeExCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", ActivationGrad{"_backward_Activation"})
.add_arguments(ActivationParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Activation)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    int act_type = dmlc::get<ActivationParam>(attrs.parsed).act_type;
    // for ReLU activation, the backward pass only needs ograd and output
    if (act_type == activation::kReLU) return 2;
    return 3;
  })
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_MKLDNN == 1
.set_attr<FInferStorageType>("FInferStorageType", BackwardActStorageType)
#endif
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
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", ActivationGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", ActivationGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
