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
 * \file dnnl_softmax.cc
 * \brief
 * \author Da Zheng
 */

#if MXNET_USE_ONEDNN == 1
#include "./dnnl_softmax-inl.h"

namespace mxnet {
namespace op {

bool SupportDNNLSoftmax(const SoftmaxParam& param, const NDArray& data, const NDArray& output) {
  const int ndim      = data.shape().ndim();
  const int in_size   = data.shape().Size();
  const int in_dtype  = data.dtype();
  const int out_dtype = output.dtype();
  const int axis      = CheckAxis(param.axis, ndim);

  if (param.temperature.has_value() && param.temperature.value() == 0.0) {
    return false;
  }

  if (in_dtype != mshadow::kFloat32 || in_dtype != out_dtype || axis != (ndim - 1)) {
    return false;
  }

  // Supports ndim up to 6
  return (ndim >= 1 && ndim <= 6 && in_size != 0);
}

void DNNLSoftmaxForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const NDArray& in_data,
                        const OpReqType& req,
                        const NDArray& out_data) {
  if (req == kNullOp)
    return;
  // Same as the FCompute path, softmax only supports kWriteTo and kWriteInplace for now
  CHECK_NE(req, kAddTo);

  const auto& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  if (param.temperature.has_value()) {
    TmpMemMgr::Get()->Init(ctx.requested[0]);
  }

  const bool is_train = ctx.is_train;
  const auto tensors  = DNNLSoftmaxFwd::Tensors(in_data, out_data);
  const auto& fwd     = DNNLSoftmaxFwd::GetCached(param, tensors, is_train);
  fwd.Execute(tensors);
}

typedef ParamOpSign<SoftmaxParam> DNNLSoftmaxSignature;
DNNLSoftmaxFwd& DNNLSoftmaxFwd::GetCached(const SoftmaxParam& param,
                                          const Tensors& tensors,
                                          const bool is_train) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxFwd, OpHash> fwds;
#endif

  DNNLSoftmaxSignature key(param);
  const float temperature = param.temperature.has_value() ? param.temperature.value() : 1.0f;
  const int axis          = CheckAxis(param.axis, tensors.data.shape().ndim());
  key.AddSign(axis);
  key.AddSign(is_train);
  key.AddSign(temperature);
  key.AddSign(tensors.data);
  key.AddSign(tensors.out);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLSoftmaxFwd fwd(param, tensors, is_train);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

softmax_fwd_pd_t DNNLSoftmaxFwd::GetSoftmaxFwdPd(const dnnl::memory& input_mem,
                                                 const int axis,
                                                 const bool is_train) {
  const auto data_md    = input_mem.get_desc();
  const auto cpu_engine = CpuEngine::Get()->get_engine();
  const auto prop = is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_scoring;
  const auto desc = dnnl::softmax_forward::desc(prop, data_md, axis);
  return softmax_fwd_pd_t(desc, cpu_engine);
}

linear_pd_t DNNLSoftmaxFwd::GetTemperaturePd(const dnnl::memory& input_mem,
                                             const float temperature) {
  const auto data_md    = input_mem.get_desc();
  const auto cpu_engine = CpuEngine::Get()->get_engine();
  const auto desc       = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_scoring,
                                                dnnl::algorithm::eltwise_linear,
                                                data_md,
                                                1.0f / temperature,
                                                0.0f);
  return linear_pd_t(desc, cpu_engine);
}

void DNNLSoftmaxFwd::Execute(const Tensors& tensors) const {
  DNNLStream* stream = DNNLStream::Get();

  auto original_input_mem = tensors.data.GetDNNLData();
  const auto out_mem      = tensors.out.GetDNNLData(softmax_pd->dst_desc());

  dnnl::memory* softmax_input_mem;
  if (temperature_pd) {
    // check whether additional buffer is needed, when temperature parameter is being used
    if (original_input_mem->get_desc() != out_mem->get_desc()) {
      softmax_input_mem = TmpMemMgr::Get()->Alloc(original_input_mem->get_desc());
    } else {
      softmax_input_mem = const_cast<dnnl::memory*>(out_mem);
    }
    stream->RegisterPrimArgs(
        *temperature_fwd,
        {{DNNL_ARG_SRC, *original_input_mem}, {DNNL_ARG_DST, *softmax_input_mem}});
  } else {
    softmax_input_mem = const_cast<dnnl::memory*>(original_input_mem);
  }

  stream->RegisterPrimArgs(*softmax_fwd,
                           {{DNNL_ARG_SRC, *softmax_input_mem}, {DNNL_ARG_DST, *out_mem}});
  stream->Submit();
}

softmax_bwd_pd_t DNNLSoftmaxBwd::GetSoftmaxBwdPd(const dnnl::memory& out_grad_mem,
                                                 const dnnl::memory& out_mem,
                                                 const int axis,
                                                 const softmax_fwd_pd_t& hint_fwd_pd) {
  dnnl::memory::desc out_grad_md = out_grad_mem.get_desc();
  dnnl::memory::desc out_md      = out_mem.get_desc();
  const auto cpu_engine          = CpuEngine::Get()->get_engine();
  const auto desc                = dnnl::softmax_backward::desc(out_grad_md, out_md, axis);
  return softmax_bwd_pd_t(desc, cpu_engine, hint_fwd_pd);
}

void DNNLSoftmaxBackward(const nnvm::NodeAttrs& attrs,
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

  const auto tensors = DNNLSoftmaxBwd::Tensors(inputs, outputs);
  const auto& bwd    = DNNLSoftmaxBwd::GetCached(param, tensors);
  bwd.Execute(tensors, req);
}

DNNLSoftmaxBwd& DNNLSoftmaxBwd::GetCached(const SoftmaxParam& param, const Tensors& tensors) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxBwd, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxBwd, OpHash> bwds;
#endif

  const float temperature = param.temperature.has_value() ? param.temperature.value() : 1.0f;
  const int axis          = CheckAxis(param.axis, tensors.out.shape().ndim());
  DNNLSoftmaxSignature key(param);
  key.AddSign(axis);
  key.AddSign(tensors.out);
  key.AddSign(tensors.out_grad);
  key.AddSign(temperature);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    DNNLSoftmaxBwd bwd(param, tensors);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void DNNLSoftmaxBwd::Execute(const Tensors& tensors, const std::vector<OpReqType>& req) const {
  DNNLStream* stream = DNNLStream::Get();

  const auto original_out_grad_mem = tensors.out_grad.GetDNNLData();
  const auto out_mem               = tensors.out.GetDNNLData();
  const auto data_grad_mem =
      CreateDNNLMem(tensors.data_grad, softmax_bwd_pd->diff_src_desc(), req[0]);

  dnnl::memory* out_grad_mem;
  if (temperature_fwd) {
    // check whether additional buffer is needed, when temperature parameter is being used
    if (original_out_grad_mem->get_desc() != softmax_bwd_pd->diff_src_desc()) {
      out_grad_mem = TmpMemMgr::Get()->Alloc(original_out_grad_mem->get_desc());
    } else {
      out_grad_mem = const_cast<dnnl::memory*>(data_grad_mem.second);
    }
    stream->RegisterPrimArgs(
        *temperature_fwd, {{DNNL_ARG_SRC, *original_out_grad_mem}, {DNNL_ARG_DST, *out_grad_mem}});
  } else {
    out_grad_mem = const_cast<dnnl::memory*>(original_out_grad_mem);
  }

  dnnl_args_map_t args = {{DNNL_ARG_DST, *out_mem},
                          {DNNL_ARG_DIFF_DST, *out_grad_mem},
                          {DNNL_ARG_DIFF_SRC, *data_grad_mem.second}};

  stream->RegisterPrimArgs(*softmax_bwd, args);

  CommitOutput(tensors.data_grad, data_grad_mem);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
