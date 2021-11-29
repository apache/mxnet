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
 * \file amp_cast.cc
 * \brief Casts used by AMP
 */

#include "./amp_cast.h"
#include "../../common/alm.h"

namespace mxnet {
namespace op {

static bool MCastChangeLayout(nnvm::NodeAttrs* attrs,
                              mshadow::LayoutFlag targetLayout,
                              std::vector<alm::Transpose>* inpTransposes,
                              std::vector<alm::Transpose>* outTransposes) {
  auto n_inps = attrs->op->get_num_inputs(*attrs);
  auto n_outs = attrs->op->get_num_outputs(*attrs);
  CHECK_EQ(n_inps, n_outs) << "This operator should have the same number inputs and outputs";
  CHECK_EQ(inpTransposes->size(), n_inps);
  CHECK_EQ(targetLayout, mshadow::kUNKNOWN);
  *outTransposes = std::move(*inpTransposes);
  inpTransposes->assign(n_inps, alm::Transpose());
  return false;
}

DMLC_REGISTER_PARAMETER(AMPCastParam);
DMLC_REGISTER_PARAMETER(AMPMultiCastParam);

#if MXNET_USE_ONEDNN == 1
static void AMPCastExCPU(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kWriteInplace) {
    return;
  }
  auto data = inputs[0];
  if (data.dtype() != mshadow::kFloat16 && outputs[0].dtype() != mshadow::kFloat16) {
    dnnl::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
    if (data.IsView() && data.IsDNNLData())
      data = data.Reorder2Default();
    const auto i_mem          = data.GetDNNLData();
    const size_t i_ndim       = data.shape().ndim();
    dnnl::memory::dims i_dims = dnnl::memory::dims(i_ndim);
    for (size_t i = 0; i < i_ndim; i++) {
      i_dims[i] = static_cast<int>(data.shape()[i]);
    }
    const auto o_desc =
        dnnl::memory::desc(i_dims,
                           get_dnnl_type(outputs[0].dtype()),
                           static_cast<dnnl::memory::format_tag>(GetDefaultFormat(i_ndim)));
    const auto out_mem = CreateDNNLMem(outputs[0], o_desc, req[0]);
    dnnl_args_map_t reorder_args;
    reorder_args[DNNL_ARG_SRC] = *i_mem;
    reorder_args[DNNL_ARG_DST] = *out_mem.second;
    DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(*i_mem, *out_mem.second), reorder_args);
    DNNLStream::Get()->Submit();
    return;
  }
  FallBackCompute(AMPCastCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool AMPCastStorageType(const nnvm::NodeAttrs& attrs,
                                      const int dev_mask,
                                      DispatchMode* dispatch_mode,
                                      std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  auto ret = DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
  return ret;
}

static void AMPMultiCastExCPU(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_outputs);
  CHECK_EQ(outputs.size(), param.num_outputs);
  dnnl::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  for (int i = 0; i < param.num_outputs; ++i) {
    if (req[i] == kWriteInplace) {
      continue;
    }
    auto data = inputs[i];
    if (data.IsView() && data.IsDNNLData())
      data = data.Reorder2Default();
    const auto i_mem          = data.GetDNNLData();
    const size_t i_ndim       = data.shape().ndim();
    dnnl::memory::dims i_dims = dnnl::memory::dims(i_ndim);
    for (size_t j = 0; j < i_ndim; j++) {
      i_dims[j] = static_cast<int>(data.shape()[j]);
    }
    const auto o_desc =
        dnnl::memory::desc(i_dims,
                           get_dnnl_type(outputs[i].dtype()),
                           static_cast<dnnl::memory::format_tag>(GetDefaultFormat(i_ndim)));
    const auto out_mem = CreateDNNLMem(outputs[i], o_desc, req[i]);
    dnnl_args_map_t reorder_args;
    reorder_args[DNNL_ARG_SRC] = *i_mem;
    reorder_args[DNNL_ARG_DST] = *out_mem.second;
    DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(*i_mem, *out_mem.second), reorder_args);
  }
  DNNLStream::Get()->Submit();
}

inline static bool AMPMultiCastStorageType(const nnvm::NodeAttrs& attrs,
                                           const int dev_mask,
                                           DispatchMode* dispatch_mode,
                                           std::vector<int>* in_attrs,
                                           std::vector<int>* out_attrs) {
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_outputs);
  CHECK_EQ(out_attrs->size(), param.num_outputs);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

#endif  // MXNET_USE_ONEDNN == 1

NNVM_REGISTER_OP(amp_cast)
    .add_alias("_npi_amp_cast")
    .describe(R"code(Cast function between low precision float/FP32 used by AMP.

It casts only between low precision float/FP32 and does not do anything for other types.
)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<AMPCastParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", AMPCastType)
    .set_attr<mxnet::alm::FChangeLayout>("FChangeLayout", ElemwiseChangeLayout)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int>>{{0, 0}};
                                    })
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<bool>{true};
                                      })
    .set_attr<FCompute>("FCompute<cpu>", AMPCastCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FInferStorageType>("FInferStorageType", AMPCastStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AMPCastExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_amp_cast"})
    .add_argument("data", "NDArray-or-Symbol", "The input.")
    .add_arguments(AMPCastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_amp_cast)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int>>{{0, 0}};
                                    })
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<bool>{true};
                                      })
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FInferStorageType>("FInferStorageType", AMPCastStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AMPCastExCPU)
#endif
    .set_attr<FCompute>("FCompute<cpu>", AMPCastCompute<cpu>);

NNVM_REGISTER_OP(amp_multicast)
    .add_alias("_npi_amp_multicast")
    .describe(R"code(Cast function used by AMP, that casts its inputs to the common widest type.

It casts only between low precision float/FP32 and does not do anything for other types.

)code" ADD_FILELINE)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_outputs);
    })
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_outputs);
    })
    .set_attr_parser(ParamParser<AMPMultiCastParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", AMPMultiCastShape)
    .set_attr<nnvm::FInferType>("FInferType", AMPMultiCastType)
    .set_attr<mxnet::alm::FChangeLayout>("FChangeLayout", MCastChangeLayout)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       uint32_t num_args =
                                           dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                       std::vector<std::string> ret;
                                       for (uint32_t i = 0; i < num_args; ++i) {
                                         ret.push_back(std::string("data_") + std::to_string(i));
                                       }
                                       return ret;
                                     })
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      int num_args =
                                          dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                      std::vector<std::pair<int, int>> ret;
                                      ret.reserve(num_args);
                                      for (int i = 0; i < num_args; ++i) {
                                        ret.emplace_back(i, i);
                                      }
                                      return ret;
                                    })
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        int num_args =
                                            dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                        return std::vector<bool>(num_args, true);
                                      })
    .set_attr<FCompute>("FCompute<cpu>", AMPMultiCastCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FInferStorageType>("FInferStorageType", AMPMultiCastStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AMPMultiCastExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_amp_multicast"})
    .add_argument("data", "NDArray-or-Symbol[]", "Weights")
    .add_arguments(AMPMultiCastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_amp_multicast)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_outputs);
    })
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_outputs);
    })
    .set_attr_parser(ParamParser<AMPMultiCastParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       uint32_t num_args =
                                           dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                       std::vector<std::string> ret;
                                       for (uint32_t i = 0; i < num_args; ++i) {
                                         ret.push_back(std::string("grad_") + std::to_string(i));
                                       }
                                       return ret;
                                     })
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      int num_args =
                                          dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                      std::vector<std::pair<int, int>> ret;
                                      ret.reserve(num_args);
                                      for (int i = 0; i < num_args; ++i) {
                                        ret.emplace_back(i, i);
                                      }
                                      return ret;
                                    })
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        int num_args =
                                            dmlc::get<AMPMultiCastParam>(attrs.parsed).num_outputs;
                                        return std::vector<bool>(num_args, true);
                                      })
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FInferStorageType>("FInferStorageType", AMPMultiCastStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", AMPMultiCastExCPU)
#endif
    .set_attr<FCompute>("FCompute<cpu>", AMPMultiCastCompute<cpu>)
    .add_argument("grad", "NDArray-or-Symbol[]", "Gradients")
    .add_arguments(AMPMultiCastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
