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
 * \file dnnl_act.cc
 * \brief
 * \author Da Zheng
 */

#if MXNET_USE_ONEDNN == 1

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "operator/operator_common.h"
#include "dnnl_base-inl.h"
#include "dnnl_act-inl.h"

namespace mxnet {
namespace op {

bool SupportDNNLAct(const ActivationParam& param) {
  return param.act_type == activation::kReLU || param.act_type == activation::kSigmoid ||
         param.act_type == activation::kLogSigmoid || param.act_type == activation::kMish ||
         param.act_type == activation::kSoftReLU || param.act_type == activation::kTanh;
}

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_eltwise.html
bool SupportDNNLAct(const ActivationParam& param, const NDArray& input) {
  return SupportDNNL<DNNLTypeMode::FloatTypes>(input) && SupportDNNLAct(param);
}

bool SupportDNNLLeakyRelu(const LeakyReLUParam& param) {
  return param.act_type == leakyrelu::kLeakyReLU || param.act_type == leakyrelu::kELU ||
         param.act_type == leakyrelu::kGELU_ERF || param.act_type == leakyrelu::kGELU_TANH;
}

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_eltwise.html
bool SupportDNNLLeakyRelu(const LeakyReLUParam& param, const NDArray& input) {
  return SupportDNNL<DNNLTypeMode::FloatTypes>(input) && SupportDNNLLeakyRelu(param);
}

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_eltwise.html
bool SupportDNNLQuantizedAct(const ActivationParam& param) {
  // Although it is the same as SupportDNNLAct i left it here, so when new activations
  // will be introduced it will be easier to handle.
  return SupportDNNLAct(param);
}

dnnl::algorithm GetDNNLActAlgo(const ActivationParam& param) {
  switch (param.act_type) {
    case activation::kReLU:
      return dnnl::algorithm::eltwise_relu;
    case activation::kSigmoid:
      return dnnl::algorithm::eltwise_logistic;
    case activation::kLogSigmoid:
      return dnnl::algorithm::eltwise_logsigmoid;
    case activation::kMish:
      return dnnl::algorithm::eltwise_mish;
    case activation::kTanh:
      return dnnl::algorithm::eltwise_tanh;
    case activation::kSoftReLU:
      return dnnl::algorithm::eltwise_soft_relu;
    default:
      LOG(FATAL) << "unknown activation type";
      return dnnl::algorithm::eltwise_relu;
  }
}

dnnl::algorithm GetDNNLActAlgo(const LeakyReLUParam& param) {
  switch (param.act_type) {
    case leakyrelu::kLeakyReLU:
      return dnnl::algorithm::eltwise_relu;
    case leakyrelu::kELU:
      return dnnl::algorithm::eltwise_elu;
    case leakyrelu::kGELU_ERF:
      return dnnl::algorithm::eltwise_gelu_erf;
    case leakyrelu::kGELU_TANH:
      return dnnl::algorithm::eltwise_gelu_tanh;
    default:
      LOG(FATAL) << "unknown activation type for LeakyReLU: " << param.act_type;
      return dnnl::algorithm::eltwise_relu;
  }
}

dnnl::eltwise_forward::primitive_desc GetActFwdDescImpl(const DNNLActParam& param,
                                                        bool is_train,
                                                        const dnnl::memory& input_mem) {
  dnnl::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine            = CpuEngine::Get()->get_engine();
  auto alg                   = param.alg;

  auto prop = is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_scoring;
  auto desc = dnnl::eltwise_forward::desc(prop, alg, data_md, param.slope);
  return dnnl::eltwise_forward::primitive_desc(desc, cpu_engine);
}

const inline dnnl::eltwise_forward& DNNLActForward::GetFwd() const {
  return *fwd_;
}

DNNLActForward& GetActForward(const DNNLActParam& param,
                              const OpContext& ctx,
                              const NDArray& in_data,
                              const dnnl::memory& in_mem) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLActSignature, DNNLActForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLActSignature, DNNLActForward, OpHash> fwds;
#endif
  DNNLActSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(static_cast<int>(param.alg));
  key.AddSign(param.slope);
  key.AddSign(in_data);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLActForward fwd(param, ctx.is_train, in_data, in_mem);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLActivationForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const NDArray& in_data,
                           const OpReqType& req,
                           const NDArray& out_data) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  DNNLActParam param_;
  param_.alg               = GetDNNLActAlgo(param);
  const NDArray& in_buffer = in_data;
  DNNLStream* stream       = DNNLStream::Get();
  auto input_mem           = in_buffer.GetDNNLData();
  DNNLActForward& fwd      = GetActForward(param_, ctx, in_buffer, *input_mem);
  auto out_mem_t           = CreateDNNLMem(out_data, fwd.fwd_pd.dst_desc(), req, &in_buffer);
  stream->RegisterPrimArgs(fwd.GetFwd(),
                           {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DST, *out_mem_t.second}});
  CommitOutput(out_data, out_mem_t);
  stream->Submit();
}

void DNNLLeakyReluForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const NDArray& in_data,
                          const OpReqType& req,
                          const NDArray& out_data) {
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  DNNLActParam param_;
  param_.alg   = GetDNNLActAlgo(param);
  param_.slope = param.slope;

  NDArray in_buffer  = in_data;
  DNNLStream* stream = DNNLStream::Get();

  if (in_data.IsView() && in_data.IsDNNLData())
    in_buffer = in_data.Reorder2Default();

  auto input_mem      = in_buffer.GetDNNLData();
  DNNLActForward& fwd = GetActForward(param_, ctx, in_buffer, *input_mem);
  auto out_mem_t      = CreateDNNLMem(out_data, fwd.fwd_pd.dst_desc(), req, &in_buffer);
  stream->RegisterPrimArgs(fwd.GetFwd(),
                           {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DST, *out_mem_t.second}});
  CommitOutput(out_data, out_mem_t);
  stream->Submit();
}

dnnl::eltwise_backward::primitive_desc GetActBwdDescImpl(const DNNLActParam& param,
                                                         const dnnl::memory& input_mem,
                                                         const dnnl::memory& diff_dst_memory) {
  dnnl::memory::desc data_md = input_mem.get_desc();
  dnnl::memory::desc diff_md = diff_dst_memory.get_desc();
  auto cpu_engine            = CpuEngine::Get()->get_engine();
  auto alg                   = param.alg;

  dnnl::eltwise_forward::desc fw_desc(dnnl::prop_kind::forward_training, alg, data_md, param.slope);
  dnnl::eltwise_forward::primitive_desc fw_pdesc(fw_desc, cpu_engine);
  dnnl::eltwise_backward::desc bw_desc(alg, diff_md, data_md, param.slope);
  dnnl::eltwise_backward::primitive_desc bw_pdesc(bw_desc, cpu_engine, fw_pdesc);
  return bw_pdesc;
}

const inline dnnl::eltwise_backward& DNNLActBackward::GetBwd() const {
  return *bwd_prim_;
}

static inline DNNLActBackward& GetActBackward(const DNNLActParam& param,
                                              const OpContext& ctx,
                                              const NDArray& in_data,
                                              const NDArray& out_grad,
                                              const dnnl::memory& in_mem) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLActSignature, DNNLActBackward, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLActSignature, DNNLActBackward, OpHash> bwds;
#endif
  DNNLActSignature key(param);
  key.AddSign(in_data);
  key.AddSign(out_grad);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    DNNLActBackward bwd(param, in_data, in_mem, *out_grad.GetDNNLData());
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

// For backward relu activation, it's okay to pass "out_data" as "in_data" to this
// function, since the computation only involes non-zeros.
void DNNLActivationBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  // XXX: for y = relu(x), y is passed as "in_data" to Backward()
  const bool relu           = param.act_type == activation::kReLU;
  const NDArray& out_buffer = inputs[0];
  const NDArray& in_buffer  = relu ? inputs[1] : inputs[2];
  const NDArray& in_grad    = outputs[0];
  DNNLActParam param_;
  param_.alg = GetDNNLActAlgo(param);
  TmpMemMgr::Get()->Init(ctx.requested[activation::kTempSpace]);
  auto diff_dst_memory = out_buffer.GetDNNLData();
  auto input_mem       = in_buffer.GetDNNLData();
  // We need to make sure the two inputs to eltwise_backward has the same memory
  // descriptor. Otherwise, the perf will suffer.
  auto diff_dst_desc = diff_dst_memory->get_desc();
  if (input_mem->get_desc() != diff_dst_desc) {
    input_mem = in_buffer.GetDNNLDataReorder(&diff_dst_desc);
  }

  DNNLActBackward& bwd = GetActBackward(param_, ctx, in_buffer, out_buffer, *input_mem);
  DNNLStream* stream   = DNNLStream::Get();
  dnnl_args_map_t args = {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DIFF_DST, *diff_dst_memory}};
  if (req[0] != kAddTo) {
    // req[0] is kWriteTo or kWriteInplace
    auto bwd_pd_diff_src_desc = bwd.bwd_pd.diff_src_desc();
    auto diff_src_memory      = const_cast<NDArray&>(in_grad).CreateDNNLData(&bwd_pd_diff_src_desc);
    args.insert({DNNL_ARG_DIFF_SRC, *diff_src_memory});
    stream->RegisterPrimArgs(bwd.GetBwd(), args);
    stream->Submit();
  } else {
    auto diff_src_memory = CreateDNNLMem(in_grad, bwd.bwd_pd.diff_src_desc(), req[0]);
    args.insert({DNNL_ARG_DIFF_SRC, *diff_src_memory.second});
    stream->RegisterPrimArgs(bwd.GetBwd(), args);
    CommitOutput(in_grad, diff_src_memory);
    stream->Submit();
  }
}

void DNNLLeakyReluBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  const NDArray& out_buffer = inputs[0];
  const NDArray& in_buffer  = inputs[1];
  const NDArray& output     = outputs[0];

  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  DNNLActParam param_;
  param_.alg   = GetDNNLActAlgo(param);
  param_.slope = param.slope;

  TmpMemMgr::Get()->Init(ctx.requested[leakyrelu::kRandom]);
  auto diff_dst_memory = out_buffer.GetDNNLData();
  auto input_mem       = in_buffer.GetDNNLData();
  // We need to make sure the two inputs to eltwise_backward has the same memory
  // descriptor. Otherwise, the perf will suffer.
  auto diff_dst_desc = diff_dst_memory->get_desc();
  if (input_mem->get_desc() != diff_dst_desc)
    input_mem = in_buffer.GetDNNLDataReorder(&diff_dst_desc);
  DNNLActBackward& bwd          = GetActBackward(param_, ctx, in_buffer, out_buffer, *input_mem);
  DNNLStream* stream            = DNNLStream::Get();
  dnnl_output_t diff_src_memory = CreateDNNLMem(output, bwd.bwd_pd.diff_src_desc(), req[0]);
  dnnl_args_map_t args          = {
      {DNNL_ARG_SRC, *input_mem},
      {DNNL_ARG_DIFF_DST, *diff_dst_memory},
      {DNNL_ARG_DIFF_SRC, *diff_src_memory.second},
  };
  stream->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(output, diff_src_memory);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
