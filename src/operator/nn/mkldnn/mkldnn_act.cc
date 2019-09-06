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

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>

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

bool SupportMKLDNNLeakyRelu(const LeakyReLUParam& param) {
  return param.act_type == leakyrelu::kLeakyReLU
      || param.act_type == leakyrelu::kELU;
}

bool SupportMKLDNNLeakyRelu(const LeakyReLUParam& param, const NDArray &input) {
  // MKL-DNN Activation supports 1d, 2d, 3d, 4d data layout
  if ((input.shape().ndim() < 1) ||
      (input.shape().ndim() > 4) ||
      (input.dtype() != mshadow::kFloat32))
    return false;
  return SupportMKLDNNLeakyRelu(param);
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

mkldnn::algorithm GetMKLDNNActAlgo(const LeakyReLUParam& param) {
  switch (param.act_type) {
    case leakyrelu::kLeakyReLU:
      return mkldnn::algorithm::eltwise_relu;
    case leakyrelu::kELU:
      return mkldnn::algorithm::eltwise_elu;
    default:
      LOG(FATAL) << "unknown activation type for LeakyReLU: " << param.act_type;
      return mkldnn::algorithm::eltwise_relu;
  }
}

mkldnn::eltwise_forward::primitive_desc GetActFwdDescImpl(
    const MKLDNNActParam& param, bool is_train,
    const mkldnn::memory &input_mem, int dtype) {
  mkldnn::memory::primitive_desc data_mpd = input_mem.get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  auto cpu_engine = data_mpd.get_engine();

  auto alg = param.alg;

  auto prop = is_train ? mkldnn::prop_kind::forward_training :
                         mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::eltwise_forward::desc(prop, alg, data_md, param.slope);
  return mkldnn::eltwise_forward::primitive_desc(desc, cpu_engine);
}

void MKLDNNActForward::SetNewMem(const mkldnn::memory &data, const mkldnn::memory &output) {
  if (this->data_ == nullptr)
    this->data_ = std::make_shared<mkldnn::memory>(data.get_primitive_desc(),
            data.get_data_handle());
  else
    this->data_->set_data_handle(data.get_data_handle());

  CHECK(fwd_pd.dst_primitive_desc() == output.get_primitive_desc());
  if (this->out_ == nullptr)
    this->out_ = std::make_shared<mkldnn::memory>(fwd_pd.dst_primitive_desc(),
            output.get_data_handle());
  else
    this->out_->set_data_handle(output.get_data_handle());

  if (this->fwd_ == nullptr) {
    this->fwd_ = std::shared_ptr<mkldnn::eltwise_forward>(
        new mkldnn::eltwise_forward(fwd_pd, mkldnn::primitive::at(*this->data_),
                                    *this->out_));
  }
}

const mkldnn::eltwise_forward &MKLDNNActForward::GetFwd() const {
  return *fwd_;
}

MKLDNNActForward &GetActForward(const MKLDNNActParam& param,
                                const OpContext &ctx, const NDArray &in_data,
                                const mkldnn::memory &in_mem) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNActSignature, MKLDNNActForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNActSignature, MKLDNNActForward, OpHash> fwds;
#endif
  MKLDNNActSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(param.alg);
  key.AddSign(param.slope);
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
  MKLDNNActParam param_;
  param_.alg = GetMKLDNNActAlgo(param);

  NDArray in_buffer = in_data;
  MKLDNNStream *stream = MKLDNNStream::Get();

  if (in_data.IsView() && in_data.IsMKLDNNData())
    in_buffer = in_data.Reorder2Default();

  auto input_mem = in_buffer.GetMKLDNNData();
  MKLDNNActForward &fwd = GetActForward(param_, ctx, in_buffer, *input_mem);
  auto out_mem_t = CreateMKLDNNMem(out_data, fwd.fwd_pd.dst_primitive_desc(), req, &in_buffer);
  fwd.SetNewMem(*input_mem, *out_mem_t.second);
  stream->RegisterPrim(fwd.GetFwd());
  CommitOutput(out_data, out_mem_t);
  stream->Submit();
}

void MKLDNNLeakyReluForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const NDArray &in_data, const OpReqType &req,
                             const NDArray &out_data) {
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  MKLDNNActParam param_;
  param_.alg = GetMKLDNNActAlgo(param);
  param_.slope = param.slope;

  NDArray in_buffer = in_data;
  MKLDNNStream *stream = MKLDNNStream::Get();

  if (in_data.IsView() && in_data.IsMKLDNNData())
    in_buffer = in_data.Reorder2Default();

  auto input_mem = in_buffer.GetMKLDNNData();
  MKLDNNActForward &fwd = GetActForward(param_, ctx, in_buffer, *input_mem);
  auto out_mem_t = CreateMKLDNNMem(out_data, fwd.fwd_pd.dst_primitive_desc(), req, &in_buffer);
  fwd.SetNewMem(*input_mem, *out_mem_t.second);
  stream->RegisterPrim(fwd.GetFwd());
  CommitOutput(out_data, out_mem_t);
  stream->Submit();
}

static mkldnn::eltwise_backward::primitive_desc GetActBwdDescImpl(
    const MKLDNNActParam &param, const mkldnn::memory &input_mem,
    const mkldnn::memory &diff_dst_memory, int dtype) {
  mkldnn::memory::primitive_desc data_mpd = input_mem.get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  mkldnn::memory::desc diff_md = diff_dst_memory.get_primitive_desc().desc();
  auto cpu_engine = data_mpd.get_engine();
  auto alg = param.alg;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    mkldnn::eltwise_forward::desc fw_desc(mkldnn::prop_kind::forward_training,
                                          alg, data_md, param.slope);
    mkldnn::eltwise_forward::primitive_desc fw_pdesc(fw_desc, cpu_engine);
    mkldnn::eltwise_backward::desc bw_desc(alg, diff_md, data_md, param.slope);
    mkldnn::eltwise_backward::primitive_desc bw_pdesc(bw_desc, cpu_engine,
                                                      fw_pdesc);
    return bw_pdesc;
  });
  LOG(FATAL) << "Unsupported data type for MKLDNN activation";
  mkldnn::eltwise_forward::desc fw_desc(mkldnn::prop_kind::forward_training,
                                        alg, data_md, param.slope);
  mkldnn::eltwise_forward::primitive_desc fw_pdesc(fw_desc, cpu_engine);
  mkldnn::eltwise_backward::desc bw_desc(alg, diff_md, data_md, param.slope);
  mkldnn::eltwise_backward::primitive_desc bw_pdesc(bw_desc, cpu_engine,
                                                    fw_pdesc);
  return bw_pdesc;
}

class MKLDNNActBackward {
  std::shared_ptr<mkldnn::eltwise_backward> bwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> diff_dst_memory;
  std::shared_ptr<mkldnn::memory> diff_src_memory;

 public:
  const mkldnn::eltwise_backward::primitive_desc pd;

  explicit MKLDNNActBackward(const MKLDNNActParam &param, const NDArray &data,
                             const mkldnn::memory &mem,
                             const mkldnn::memory &diff_dst_memory)
      : pd(GetActBwdDescImpl(param, mem, diff_dst_memory, data.dtype())) {}

  void SetNewMem(const mkldnn::memory &data,
                 const mkldnn::memory &diff_dst_memory,
                 const mkldnn::memory &diff_src_memory) {
    if (this->bwd != nullptr) {
      this->data->set_data_handle(data.get_data_handle());
      this->diff_dst_memory->set_data_handle(diff_dst_memory.get_data_handle());
      this->diff_src_memory->set_data_handle(diff_src_memory.get_data_handle());
    } else {
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          data.get_primitive_desc(), data.get_data_handle()));
      this->diff_dst_memory = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(diff_dst_memory.get_primitive_desc(),
                             diff_dst_memory.get_data_handle()));
      this->diff_src_memory = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(diff_src_memory.get_primitive_desc(),
                             diff_src_memory.get_data_handle()));
      this->bwd = std::shared_ptr<mkldnn::eltwise_backward>(
          new mkldnn::eltwise_backward(
              this->pd, mkldnn::primitive::at(*this->data),
              *this->diff_dst_memory, *this->diff_src_memory));
    }
  }

  const inline mkldnn::eltwise_backward &GetBwd() const { return *bwd; }
};

static inline MKLDNNActBackward &GetActBackward(const MKLDNNActParam &param,
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
  MKLDNNActParam param_;
  param_.alg = GetMKLDNNActAlgo(param);
  TmpMemMgr::Get()->Init(ctx.requested[activation::kTempSpace]);
  auto diff_dst_memory = out_buffer.GetMKLDNNData();
  auto input_mem = in_buffer.GetMKLDNNData();
  // We need to make sure the two inputs to eltwise_backward has the same memory
  // descriptor. Otherwise, the perf will suffer.
  if (input_mem->get_primitive_desc() != diff_dst_memory->get_primitive_desc())
    input_mem = in_buffer.GetMKLDNNDataReorder(diff_dst_memory->get_primitive_desc());
  MKLDNNActBackward &bwd =
      GetActBackward(param_, ctx, in_buffer, out_buffer, *input_mem);
  MKLDNNStream *stream = MKLDNNStream::Get();
  mkldnn_output_t diff_src_memory =
      CreateMKLDNNMem(in_grad, bwd.pd.diff_src_primitive_desc(), req);
  bwd.SetNewMem(*input_mem, *diff_dst_memory, *diff_src_memory.second);
  stream->RegisterPrim(bwd.GetBwd());
  CommitOutput(in_grad, diff_src_memory);
  stream->Submit();
}

void MKLDNNLeakyReluBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext &ctx,
                             const std::vector<NDArray>& inputs,
                             const OpReqType &req,
                             const NDArray &output) {
  if (req == kNullOp) {
    return;
  }
  CHECK_GE(inputs.size(), 2U);
  NDArray out_buffer = inputs[0];
  if (inputs[0].IsView() && inputs[0].IsMKLDNNData())
    out_buffer = inputs[0].Reorder2Default();

  NDArray in_buffer = inputs[1];
  if (inputs[1].IsView() && inputs[1].IsMKLDNNData())
    in_buffer = inputs[1].Reorder2Default();

  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  MKLDNNActParam param_;
  param_.alg = GetMKLDNNActAlgo(param);
  param_.slope = param.slope;

  TmpMemMgr::Get()->Init(ctx.requested[leakyrelu::kRandom]);
  auto diff_dst_memory = out_buffer.GetMKLDNNData();
  auto input_mem = in_buffer.GetMKLDNNData();
  // We need to make sure the two inputs to eltwise_backward has the same memory
  // descriptor. Otherwise, the perf will suffer.
  if (input_mem->get_primitive_desc() != diff_dst_memory->get_primitive_desc())
    input_mem = in_buffer.GetMKLDNNDataReorder(diff_dst_memory->get_primitive_desc());
  MKLDNNActBackward &bwd =
      GetActBackward(param_, ctx, in_buffer, out_buffer, *input_mem);
  MKLDNNStream *stream = MKLDNNStream::Get();
  mkldnn_output_t diff_src_memory =
      CreateMKLDNNMem(output, bwd.pd.diff_src_primitive_desc(), req);
  bwd.SetNewMem(*input_mem, *diff_dst_memory, *diff_src_memory.second);
  stream->RegisterPrim(bwd.GetBwd());
  CommitOutput(output, diff_src_memory);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif
