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
  // MKLDNN does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
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
  int real_axis     = CheckAxis(param.axis, tensors.data.shape().ndim());
  key.AddSign(real_axis);
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

static mkldnn::softmax_backward::primitive_desc GetSoftmaxBwdPd(
    const mkldnn::memory& diff_mem,
    const mkldnn::memory& data_mem,
    const int axis,
    const mkldnn::softmax_forward::primitive_desc& hint_fwd_pd) {
  mkldnn::memory::desc diff_md = diff_mem.get_desc();
  mkldnn::memory::desc data_md = data_mem.get_desc();
  auto cpu_engine              = CpuEngine::Get()->get_engine();
  auto desc                    = mkldnn::softmax_backward::desc(diff_md, data_md, axis);
  return mkldnn::softmax_backward::primitive_desc(desc, cpu_engine, hint_fwd_pd);
}

class MKLDNNSoftmaxBwd {
 public:
  mkldnn::softmax_backward::primitive_desc softmax_bwd_pd;
  mkldnn::eltwise_forward::primitive_desc temperature_pd;

  MKLDNNSoftmaxBwd(const mkldnn::memory& diff_mem,
                   const mkldnn::memory& data_mem,
                   const int axis,
                   const double temperature,
                   const mkldnn::softmax_forward::primitive_desc& hint_fwd_pd)
      : softmax_bwd_pd(GetSoftmaxBwdPd(diff_mem, data_mem, axis, hint_fwd_pd)) {
    if (temperature != 1.0) {
      temperature_pd  = MKLDNNSoftmaxFwd::GetTemperaturePd(data_mem, temperature);
      temperature_fwd = std::make_shared<mkldnn::eltwise_forward>(temperature_pd);
    }
    softmax_bwd = std::make_shared<mkldnn::softmax_backward>(softmax_bwd_pd);
  }

  const mkldnn::eltwise_forward& GetTemperatureFwd() const {
    return *temperature_fwd;
  }
  const mkldnn::softmax_backward& GetSoftmaxBwd() const {
    return *softmax_bwd;
  }

 private:
  std::shared_ptr<mkldnn::softmax_backward> softmax_bwd;
  std::shared_ptr<mkldnn::eltwise_forward> temperature_fwd;
};

static MKLDNNSoftmaxBwd& GetSoftmaxBwd(const SoftmaxParam& param,
                                       const int real_axis,
                                       const std::vector<NDArray>& data,
                                       const std::vector<NDArray>& output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxBwd, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxBwd, OpHash> bwds;
#endif

  float temperature = param.temperature.has_value() ? param.temperature.value() : 1.0f;
  MKLDNNSoftmaxSignature key(param);
  key.AddSign(real_axis);
  key.AddSign(data);
  key.AddSign(output);
  key.AddSign(temperature);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto diff_mem       = data[0].GetMKLDNNData();
    auto data_mem       = data[1].GetMKLDNNData();
    auto softmax_fwd_pd = MKLDNNSoftmaxFwd::GetSoftmaxFwdPd(*data_mem, real_axis, true);
    MKLDNNSoftmaxBwd bwd(*diff_mem, *data_mem, real_axis, temperature, softmax_fwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& in_data,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& out_data) {
  if (req[0] == kNullOp)
    return;

  CHECK_EQ(in_data.size(), 2U);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  double temperature        = param.temperature.has_value() ? param.temperature.value() : 1.0;
  int axis                  = CheckAxis(param.axis, in_data[1].shape().ndim());
  auto original_diff_mem    = in_data[0].GetMKLDNNData();
  auto data_mem             = in_data[1].GetMKLDNNData();
  auto bwd                  = GetSoftmaxBwd(param, axis, in_data, out_data);

  auto out_mem         = CreateMKLDNNMem(out_data[0], bwd.softmax_bwd_pd.diff_src_desc(), req[0]);
  MKLDNNStream* stream = MKLDNNStream::Get();

  mkldnn::memory* diff_mem;
  if (temperature != 1.0) {
    // check whether additional buffer is needed to apply division
    if (original_diff_mem->get_desc() != bwd.softmax_bwd_pd.diff_src_desc()) {
      TmpMemMgr::Get()->Init(ctx.requested[0]);
      diff_mem = TmpMemMgr::Get()->Alloc(original_diff_mem->get_desc());
    } else {
      diff_mem = const_cast<mkldnn::memory*>(out_mem.second);
    }
    stream->RegisterPrimArgs(bwd.GetTemperatureFwd(),
                             {{MKLDNN_ARG_SRC, *original_diff_mem}, {MKLDNN_ARG_DST, *diff_mem}});
  } else {
    diff_mem = const_cast<mkldnn::memory*>(original_diff_mem);
  }

  mkldnn_args_map_t args = {{MKLDNN_ARG_DST, *data_mem},
                            {MKLDNN_ARG_DIFF_DST, *diff_mem},
                            {MKLDNN_ARG_DIFF_SRC, *out_mem.second}};

  stream->RegisterPrimArgs(bwd.GetSoftmaxBwd(), args);

  CommitOutput(out_data[0], out_mem);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
