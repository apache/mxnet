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
 * \file leaky_relu.cc
 * \brief
 * \author Bing Xu
 */

#include "./leaky_relu-inl.h"
#include "../common/alm.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_act-inl.h"
#endif  // MXNET_USE_ONEDNN == 1

#include <nnvm/op_attr_types.h>
namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LeakyReLUParam);

static bool LeakyReLUType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_type,
                          std::vector<int>* out_type) {
  int dtype = -1;
  for (const int& type : *in_type) {
    type_assign(&dtype, type);
  }
  for (const int& type : *out_type) {
    type_assign(&dtype, type);
  }
  for (size_t i = 0; i < in_type->size(); ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, dtype);
  }
  for (size_t i = 0; i < out_type->size(); ++i) {
    TYPE_ASSIGN_CHECK(*out_type, i, dtype);
  }
  return dtype != -1;
}

static bool LeakyReLUShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_shape,
                           std::vector<TShape>* out_shape) {
  using namespace mshadow;
  const LeakyReLUParam& param_ = nnvm::get<LeakyReLUParam>(attrs.parsed);
  if (param_.act_type == leakyrelu::kPReLU) {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, gamma]";
  } else {
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
  }
  const mxnet::TShape& dshape = in_shape->at(leakyrelu::kData);
  if (!mxnet::ndim_is_known(dshape))
    return false;
  if (param_.act_type == leakyrelu::kPReLU) {
    const mxnet::TShape& gshape = in_shape->at(leakyrelu::kGamma);
    if (!mxnet::ndim_is_known(gshape)) {
      in_shape->at(leakyrelu::kGamma) = mxnet::TShape(Shape1(dshape[1]));
    }
    if (dshape == gshape) {
      SHAPE_ASSIGN_CHECK(*out_shape, 0, dshape);
    }
  }
  out_shape->clear();
  out_shape->push_back(dshape);
  if (param_.act_type == leakyrelu::kRReLU) {
    out_shape->push_back(dshape);
  }
  return true;
}

#if MXNET_USE_ONEDNN == 1
static void LeakyReLUComputeExCPU(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  if (inputs[0].shape().Size() == 0U)
    return;
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  size_t expected             = param.act_type == leakyrelu::kPReLU ? 2 : 1;
  CHECK_EQ(inputs.size(), expected);
  if (SupportDNNLLeakyRelu(param, inputs[0])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLLeakyReluForward, attrs, ctx, inputs[0], req[0], outputs[0]);
    DNNL_OPCHECK_RUN(LeakyReLUCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(LeakyReLUCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

void LeakyReLUGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  if (inputs[0].shape().Size() == 0U)
    return;
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  if (SupportDNNLLeakyRelu(param, inputs[0])) {
    std::vector<NDArray> in_data{inputs[0], inputs[1]};
    DNNL_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    DNNLRun(DNNLLeakyReluBackward, attrs, ctx, in_data, req, outputs);
    DNNL_OPCHECK_RUN(LeakyReLUGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(LeakyReLUGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool LeakyReLUStorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int>* in_attrs,
                                        std::vector<int>* out_attrs) {
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  size_t expected             = param.act_type == leakyrelu::kPReLU ? 2 : 1;
  CHECK_EQ(in_attrs->size(), expected);
  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLLeakyRelu(param), dispatch_mode, in_attrs, out_attrs);
}

inline static bool BackwardLeakyReLUStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int>* in_attrs,
                                                std::vector<int>* out_attrs) {
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLLeakyRelu(param), dispatch_mode, in_attrs, out_attrs);
}
#endif  // MXNET_USE_ONEDNN == 1

static bool LRChangeLayout(nnvm::NodeAttrs* attrs,
                           mshadow::LayoutFlag target_layout,
                           std::vector<alm::Transpose>* in_axes,
                           std::vector<alm::Transpose>* out_axes) {
  CHECK_EQ(target_layout, mshadow::kUNKNOWN);
  out_axes->assign(1, alm::FactorCommonTranspose(in_axes));
  if (attrs->dict["act_type"] == "rrelu")
    out_axes->resize(2);
  return false;
}

static void LeakyReLUParamParser(nnvm::NodeAttrs* attrs) {
  // For backward compatibility, replace gelu to gelu_erf
  auto iter = attrs->dict.find("act_type");
  if (iter != attrs->dict.end()) {
    auto& type = attrs->dict["act_type"];
    if (type == "gelu") {
      type = "gelu_erf";
    }
  }
  ParamParser<LeakyReLUParam>(attrs);
}

NNVM_REGISTER_OP(LeakyReLU)
    .describe(R"code(Applies Leaky rectified linear unit activation element-wise to the input.

Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
when the input is negative and has a slope of one when input is positive.

The following modified ReLU Activation functions are supported:

- *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
- *gelu*: Gaussian Error Linear Unit. `y = 0.5 * x * (1 + erf(x / sqrt(2)))`
- *gelu_erf*: Same as gelu.
- *gelu_tanh*: Gaussian Error Linear Unit using tanh function.
  `y = 0.5 * x * (1 + tanh((sqrt(2/pi) * (x + 0.044715*x^3))))`
- *selu*: Scaled Exponential Linear Unit. `y = lambda * (x > 0 ? x : alpha * (exp(x) - 1))` where
  *lambda = 1.0507009873554804934193349852946* and *alpha = 1.6732632423543772848170429916717*.
- *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
- *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.
- *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from
  *[lower_bound, upper_bound)* for training, while fixed to be
  *(lower_bound+upper_bound)/2* for inference.

)code" ADD_FILELINE)
    .add_alias("_npx_leaky_relu")
    .set_num_inputs([](const NodeAttrs& attrs) {
      const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
      return param.act_type == leakyrelu::kPReLU ? 2 : 1;
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
      return param.act_type == leakyrelu::kRReLU ? 2 : 1;
    })
    .set_attr_parser(LeakyReLUParamParser)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", LeakyReLUStorageType)
#endif
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       const LeakyReLUParam& param =
                                           nnvm::get<LeakyReLUParam>(attrs.parsed);
                                       return param.act_type == leakyrelu::kPReLU ?
                                                  std::vector<std::string>{"data", "gamma"} :
                                                  std::vector<std::string>{"data"};
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        const LeakyReLUParam& param =
                                            nnvm::get<LeakyReLUParam>(attrs.parsed);
                                        return param.act_type == leakyrelu::kRReLU ?
                                                   std::vector<std::string>{"output", "mask"} :
                                                   std::vector<std::string>{"output"};
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", LeakyReLUShape)
    .set_attr<nnvm::FInferType>("FInferType", LeakyReLUType)
    .set_attr<mxnet::alm::FChangeLayout>("FChangeLayout", LRChangeLayout)
    .set_attr<FCompute>("FCompute<cpu>", LeakyReLUCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", LeakyReLUComputeExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_LeakyReLU"})
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("data", "NDArray-or-Symbol", "Input data to activation function.")
    .add_argument("gamma", "NDArray-or-Symbol", "Input data to activation function.")
    .add_arguments(LeakyReLUParam::__FIELDS__())
    .set_attr<nnvm::FSetInputVarAttrOnCompose>(
        "FSetInputVarAttrOnCompose",
        [](const nnvm::NodeAttrs& attrs, nnvm::ObjectPtr var, const int index) {
          if (index == 1 && var->attrs.dict.find("__init__") == var->attrs.dict.end()) {
            var->attrs.dict["__init__"] = R"(["Constant", {"value": 0.25}])";
          }
        });

NNVM_REGISTER_OP(_backward_LeakyReLU)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
      if (param.act_type == leakyrelu::kPReLU) {
        // forward has 2 inputs and 1 output
        return 2 + 2 * 1;
      } else if (param.act_type == leakyrelu::kRReLU) {
        // forward has 1 input and 2 outputs
        return 1 + 2 * 2;
      } else {
        // forward has 1 input and 1 output
        return 1 + 2 * 1;
      }
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
      return param.act_type == leakyrelu::kPReLU ? 2 : 1;
    })
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", BackwardLeakyReLUStorageType)
#endif
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr_parser(LeakyReLUParamParser)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", LeakyReLUGradComputeExCPU)
#endif
    .set_attr<FCompute>("FCompute<cpu>", LeakyReLUGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
