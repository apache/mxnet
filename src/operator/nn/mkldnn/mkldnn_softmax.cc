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
 * \file mkldnn_softmax.cc
 * \brief
 * \author Da Zheng
 */

#if MXNET_USE_ONEDNN == 1

#include "./mkldnn_softmax-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNSoftmax(const SoftmaxParam& param, const NDArray& data, const NDArray& output) {
  const int ndim      = data.shape().ndim();
  const int in_dtype  = data.dtype();
  const int out_dtype = output.dtype();
  const int axis      = CheckAxis(param.axis, ndim);

  // Currently, MKLDNN shows bad performance when softmax is not performed on the last dimension
  if (in_dtype != mshadow::kFloat32 || in_dtype != out_dtype || axis != (ndim - 1)) {
    return false;
  }

  // only supports ndim = 1, 2, 3, 4 for now
  return (ndim >= 1 && ndim <= 4);
}

void MKLDNNSoftmaxForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const NDArray& in_data,
                          const OpReqType& req,
                          const NDArray& out_data) {
  if (req == kNullOp)
    return;
  // same as the FCompute path, softmax only supports kWriteTo and kWriteInplace for now.
  CHECK_NE(req, kAddTo);

  const auto& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (param.temperature.has_value()) {
    TmpMemMgr::Get()->Init(ctx.requested[0]);
  }

  const bool is_train = ctx.is_train;
  const auto tensors  = MKLDNNSoftmaxFwd::Tensors(in_data, out_data);
  const auto& fwd     = MKLDNNSoftmaxFwd::GetCached(param, tensors, is_train);
  fwd.Execute(tensors);
}

typedef ParamOpSign<SoftmaxParam> MKLDNNSoftmaxSignature;
MKLDNNSoftmaxFwd& MKLDNNSoftmaxFwd::GetCached(const SoftmaxParam& param,
                                              const Tensors& tensors,
                                              const bool is_train) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxFwd, OpHash> fwds;
#endif

  MKLDNNSoftmaxSignature key(param);
  float temperature = param.temperature.has_value() ? param.temperature.value() : 1.0f;
  int axis          = CheckAxis(param.axis, tensors.data.shape().ndim());
  key.AddSign(axis);
  key.AddSign(is_train);
  key.AddSign(temperature);
  key.AddSign(tensors.data);
  key.AddSign(tensors.out);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSoftmaxFwd fwd(param, tensors, is_train);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

softmax_fwd_pd_t MKLDNNSoftmaxFwd::GetSoftmaxFwdPd(const mkldnn::memory& input_mem,
                                                   const int axis,
                                                   bool is_train) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine              = CpuEngine::Get()->get_engine();
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::softmax_forward::desc(prop, data_md, axis);
  return softmax_fwd_pd_t(desc, cpu_engine);
}

linear_pd_t MKLDNNSoftmaxFwd::GetTemperaturePd(const mkldnn::memory& input_mem, float temperature) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine              = CpuEngine::Get()->get_engine();
  auto desc                    = mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_scoring,
                                            mkldnn::algorithm::eltwise_linear,
                                            data_md,
                                            1.0f / temperature,
                                            0.0f);
  return linear_pd_t(desc, cpu_engine);
}

void MKLDNNSoftmaxFwd::Execute(const Tensors& tensors) const {
  MKLDNNStream* stream = MKLDNNStream::Get();

  auto original_input_mem = tensors.data.GetMKLDNNData();
  auto out_mem            = tensors.out.GetMKLDNNData(softmax_pd->dst_desc());

  mkldnn::memory* softmax_input_mem;
  if (temperature_pd) {  // temperature parameter used
    // check whether additional buffer is needed
    if (original_input_mem->get_desc() != out_mem->get_desc()) {
      softmax_input_mem = TmpMemMgr::Get()->Alloc(original_input_mem->get_desc());
    } else {
      softmax_input_mem = const_cast<mkldnn::memory*>(out_mem);
    }
    stream->RegisterPrimArgs(
        *temperature_fwd,
        {{MKLDNN_ARG_SRC, *original_input_mem}, {MKLDNN_ARG_DST, *softmax_input_mem}});
  } else {
    softmax_input_mem = const_cast<mkldnn::memory*>(original_input_mem);
  }

  stream->RegisterPrimArgs(*softmax_fwd,
                           {{MKLDNN_ARG_SRC, *softmax_input_mem}, {MKLDNN_ARG_DST, *out_mem}});
  stream->Submit();
}

softmax_bwd_pd_t MKLDNNSoftmaxBwd::GetSoftmaxBwdPd(const mkldnn::memory& out_grad_mem,
                                                   const mkldnn::memory& out_mem,
                                                   const int axis,
                                                   const softmax_fwd_pd_t& hint_fwd_pd) {
  mkldnn::memory::desc out_grad_md = out_grad_mem.get_desc();
  mkldnn::memory::desc out_md      = out_mem.get_desc();
  auto cpu_engine                  = CpuEngine::Get()->get_engine();
  auto desc                        = mkldnn::softmax_backward::desc(out_grad_md, out_md, axis);
  return softmax_bwd_pd_t(desc, cpu_engine, hint_fwd_pd);
}

void MKLDNNSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp)
    return;

  const auto& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (param.temperature.has_value()) {
    TmpMemMgr::Get()->Init(ctx.requested[0]);
  }

  const auto tensors = MKLDNNSoftmaxBwd::Tensors(inputs, outputs);
  const auto& bwd    = MKLDNNSoftmaxBwd::GetCached(param, tensors);
  bwd.Execute(tensors, req);
}

MKLDNNSoftmaxBwd& MKLDNNSoftmaxBwd::GetCached(const SoftmaxParam& param, const Tensors& tensors) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxBwd, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxBwd, OpHash> bwds;
#endif

  float temperature = param.temperature.has_value() ? param.temperature.value() : 1.0f;
  int axis          = CheckAxis(param.axis, tensors.out.shape().ndim());
  MKLDNNSoftmaxSignature key(param);
  key.AddSign(axis);
  key.AddSign(tensors.out);
  key.AddSign(tensors.out_grad);
  key.AddSign(temperature);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    MKLDNNSoftmaxBwd bwd(param, tensors);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNSoftmaxBwd::Execute(const Tensors& tensors, const std::vector<OpReqType>& req) const {
  MKLDNNStream* stream = MKLDNNStream::Get();

  auto original_out_grad_mem = tensors.out_grad.GetMKLDNNData();
  auto out_mem               = tensors.out.GetMKLDNNData();
  auto data_grad_mem = CreateMKLDNNMem(tensors.data_grad, softmax_bwd_pd->diff_src_desc(), req[0]);

  mkldnn::memory* out_grad_mem;
  if (temperature_fwd) {
    // check whether additional buffer is needed to apply division
    if (original_out_grad_mem->get_desc() != softmax_bwd_pd->diff_src_desc()) {
      out_grad_mem = TmpMemMgr::Get()->Alloc(original_out_grad_mem->get_desc());
    } else {
      out_grad_mem = const_cast<mkldnn::memory*>(data_grad_mem.second);
    }
    stream->RegisterPrimArgs(
        *temperature_fwd,
        {{MKLDNN_ARG_SRC, *original_out_grad_mem}, {MKLDNN_ARG_DST, *out_grad_mem}});
  } else {
    out_grad_mem = const_cast<mkldnn::memory*>(original_out_grad_mem);
  }

  mkldnn_args_map_t args = {{MKLDNN_ARG_DST, *out_mem},
                            {MKLDNN_ARG_DIFF_DST, *out_grad_mem},
                            {MKLDNN_ARG_DIFF_SRC, *data_grad_mem.second}};

  stream->RegisterPrimArgs(*softmax_bwd, args);

  CommitOutput(tensors.data_grad, data_grad_mem);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
