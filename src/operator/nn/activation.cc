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
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu, Da Zheng
 */
#include "./activation-inl.h"
#include "../mshadow_op.h"
#include "../tensor/elemwise_unary_op.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#endif  // MXNET_USE_ONEDNN == 1
#include "../operator_common.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

namespace activation {

int GradNumInputs(int act_type) {
  // check activation.cu \sa ActivationGradCompute
  if (dmlc::GetEnv("MXNET_MEMORY_OPT", 0)) {
    return 2;
  }
  switch (act_type) {
    case kReLU:
      return 2;
    case kSoftReLU:
    case kSoftSign:
    case kTanh:
    case kSigmoid:
    case kLogSigmoid:
    case kMish:
      return 3;
    default:
      CHECK(false) << "missing activation type";
  }
  // unreachable
  return -1;
}

}  // namespace activation

DMLC_REGISTER_PARAMETER(ActivationParam);

// This will determine the order of the inputs for backward computation.
struct ActivationGrad {
  const char* op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    // ograds
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    const NodeAttrs& attrs = n->attrs;
    using namespace activation;
    int act_type = dmlc::get<ActivationParam>(attrs.parsed).act_type;

    if (dmlc::GetEnv("MXNET_MEMORY_OPT", 0)) {
      if (act_type == kSoftSign) {
        heads.push_back(n->inputs[activation::kData]);
      } else {
        heads.emplace_back(n, activation::kOut, 0);
      }
    } else {
      heads.emplace_back(n, activation::kOut, 0);  // output
      // for ReLU, no need to pass input data. This enables inplace optimization
      // during the forward pass. check activation.cu \sa ActivationGradCompute
      switch (act_type) {
        case kReLU:
          break;
        case kSoftReLU:
        case kSoftSign:
        case kTanh:
        case kSigmoid:
        case kLogSigmoid:
        case kMish:
          heads.push_back(n->inputs[activation::kData]);
          break;
        default:
          CHECK(false) << "missing activation type";
      }
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

#if MXNET_USE_ONEDNN == 1
static void ActivationComputeExCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (SupportDNNLAct(param, inputs[0])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLActivationForward, attrs, ctx, inputs[0], req[0], outputs[0]);
    DNNL_OPCHECK_RUN(ActivationCompute<cpu>, attrs, ctx, inputs, req, outputs);
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
  CHECK_EQ(inputs.size(), activation::GradNumInputs(param.act_type));
  if (SupportDNNLAct(param, inputs[0])) {
    DNNL_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    DNNLRun(DNNLActivationBackward, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(ActivationGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(ActivationGradComputeImpl<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool ActivationStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int>* in_attrs,
                                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLAct(param), dispatch_mode, in_attrs, out_attrs);
}

inline static bool BackwardActStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), activation::GradNumInputs(param.act_type));
  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLAct(param), dispatch_mode, in_attrs, out_attrs);
}
#endif  // MXNET_USE_ONEDNN == 1

MXNET_OPERATOR_REGISTER_UNARY(Activation)
    .add_alias("_npx_activation")
    .describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `log_sigmoid`: :math:`y = log(\frac{1}{1 + exp(-x)})`
- `mish`: :math:`y = x * tanh(log(1 + exp(x)))`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
- `softsign`: :math:`y = \frac{x}{1 + abs(x)}`

)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<ActivationParam>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", ActivationStorageType)
#endif
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output"};
                                      })
    .set_attr<FCompute>("FCompute<cpu>", ActivationCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", ActivationComputeExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ActivationGrad{"_backward_Activation"})
    .add_arguments(ActivationParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Activation)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const int act_type = dmlc::get<ActivationParam>(attrs.parsed).act_type;
      return activation::GradNumInputs(act_type);
    })
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", BackwardActStorageType)
#endif
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<-1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif
    .set_attr_parser(ParamParser<ActivationParam>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", ActivationGradComputeExCPU)
#endif
    .set_attr<FCompute>("FCompute<cpu>", ActivationGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
