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
 * \file mkldnn_act.cc
 * \brief
 * \author Da Zheng
*/

#if MXNET_USE_MKLDNN == 100

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../operator_common.h"
#include "mkldnn_act-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNAct(const ActivationParam& param) {
  return param.act_type == activation::kReLU
      || param.act_type == activation::kSigmoid
      || param.act_type == activation::kSoftReLU
      || param.act_type == activation::kTanh;
}

bool SupportMKLDNNAct(const ActivationParam& param, const NDArray &input) {
  // MKL-DNN Activation supports 1d, 2d, 3d, 4d data layout
  if ((input.shape().ndim() < 1) ||
      (input.shape().ndim() > 4) ||
      (input.dtype() != mshadow::kFloat32))
    return false;
  return SupportMKLDNNAct(param);
}

bool SupportQuantizedMKLDNNAct(const ActivationParam &param) {
  // TODO(zhennan): Add more activation type when mkldnn supports.
  //                Remove this when it's identity to SupportMKLDNNAct.
  return param.act_type == activation::kReLU;
}

mkldnn::algorithm GetMKLDNNActAlgo(const ActivationParam& param) {
  switch (param.act_type) {
    case activation::kReLU:
      return mkldnn::algorithm::eltwise_relu;
    case activation::kSigmoid:
      return mkldnn::algorithm::eltwise_logistic;
    case activation::kTanh:
      return mkldnn::algorithm::eltwise_tanh;
    case activation::kSoftReLU:
      return mkldnn::algorithm::eltwise_soft_relu;
    default:
      LOG(FATAL) << "unknown activation type";
      return mkldnn::algorithm::eltwise_relu;
  }
}

mkldnn::eltwise_forward::primitive_desc GetActFwdDescImpl(
    const ActivationParam& param, bool is_train,
    const mkldnn::memory &input_mem) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto alg = GetMKLDNNActAlgo(param);

  auto prop = is_train ? mkldnn::prop_kind::forward_training :
                         mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::eltwise_forward::desc(prop, alg, data_md, 0.0f);

  return mkldnn::eltwise_forward::primitive_desc(desc, cpu_engine);
}

const inline mkldnn::eltwise_forward &MKLDNNActForward::GetFwd() const {
  return *fwd_;
}

MKLDNNActForward &GetActForward(const ActivationParam& param,
                                const OpContext &ctx, const NDArray &in_data,
                                const mkldnn::memory &in_mem) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNActSignature, MKLDNNActForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNActSignature, MKLDNNActForward, OpHash> fwds;
#endif
  MKLDNNActSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(param.act_type);
  key.AddSign(in_data);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNActForward fwd(param, ctx.is_train, in_data, in_mem);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNActivationForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const NDArray &in_data, const OpReqType &req,
                             const NDArray &out_data) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);

  NDArray in_buffer = in_data;
  MKLDNNStream *stream = MKLDNNStream::Get();

  if (in_data.IsView() && in_data.IsMKLDNNData())
    in_buffer = in_data.Reorder2Default();

  auto input_mem = in_buffer.GetMKLDNNData();
  MKLDNNActForward &fwd = GetActForward(param, ctx, in_buffer, *input_mem);
  auto out_mem_t = CreateMKLDNNMem(out_data, fwd.fwd_pd.dst_desc(), req, &in_buffer);
  stream->RegisterPrimArgs(fwd.GetFwd(),
                           {{ MKLDNN_ARG_SRC, *input_mem}, { MKLDNN_ARG_DST, *out_mem_t.second}});
  CommitOutput(out_data, out_mem_t);
  stream->Submit();
}

mkldnn::eltwise_backward::primitive_desc GetActBwdDescImpl(
    const ActivationParam &param, const mkldnn::memory &input_mem,
    const mkldnn::memory &diff_dst_memory) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  mkldnn::memory::desc diff_md = diff_dst_memory.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto alg = GetMKLDNNActAlgo(param);

  float alpha = 0;
  mkldnn::eltwise_forward::desc fw_desc(mkldnn::prop_kind::forward_training,
                                        alg, data_md, alpha);
  mkldnn::eltwise_forward::primitive_desc fw_pdesc(fw_desc, cpu_engine);
  mkldnn::eltwise_backward::desc bw_desc(alg, diff_md, data_md, alpha);
  mkldnn::eltwise_backward::primitive_desc bw_pdesc(bw_desc, cpu_engine,
                                                    fw_pdesc);
  return bw_pdesc;
}

const inline mkldnn::eltwise_backward &MKLDNNActBackward::GetBwd() const {
  return *bwd;
}

static inline MKLDNNActBackward &GetActBackward(const ActivationParam &param,
                                                const OpContext &ctx,
                                                const NDArray &in_data,
                                                const NDArray &out_grad,
                                                const mkldnn::memory &in_mem) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNActSignature, MKLDNNActBackward, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNActSignature, MKLDNNActBackward, OpHash> bwds;
#endif
  MKLDNNActSignature key(param);
  key.AddSign(in_data);
  key.AddSign(out_grad);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    MKLDNNActBackward bwd(param, in_data, in_mem, *out_grad.GetMKLDNNData());
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

// For backward relu activation, it's okay to pass "out_data" as "in_data" to this
// function, since the computation only involes non-zeros.
void MKLDNNActivationBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                              const NDArray &out_grad, const NDArray &in_data,
                              const OpReqType &req, const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }

  NDArray out_buffer = out_grad;
  if (out_grad.IsView() && out_grad.IsMKLDNNData())
    out_buffer = out_grad.Reorder2Default();

  NDArray in_buffer = in_data;
  if (in_data.IsView() && in_data.IsMKLDNNData())
    in_buffer = in_data.Reorder2Default();

  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  TmpMemMgr::Get()->Init(ctx.requested[activation::kTempSpace]);
  auto diff_dst_memory = out_buffer.GetMKLDNNData();
  auto input_mem = in_buffer.GetMKLDNNData();
  // We need to make sure the two inputs to eltwise_backward has the same memory
  // descriptor. Otherwise, the perf will suffer.
  if (input_mem->get_desc() != diff_dst_memory->get_desc())
    input_mem = in_buffer.GetMKLDNNDataReorder(diff_dst_memory->get_desc());
  MKLDNNActBackward &bwd =
      GetActBackward(param, ctx, in_buffer, out_buffer, *input_mem);
  MKLDNNStream *stream = MKLDNNStream::Get();
  mkldnn_output_t diff_src_memory =
      CreateMKLDNNMem(in_grad, bwd.pd.diff_src_desc(), req);
  mkldnn_args_map_t args = {
    { MKLDNN_ARG_SRC, *input_mem },
    { MKLDNN_ARG_DIFF_DST, *diff_dst_memory },
    { MKLDNN_ARG_DIFF_SRC, *diff_src_memory.second },
  };
  stream->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(in_grad, diff_src_memory);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
