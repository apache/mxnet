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
 * \file dnnl_softmax_output.cc
 * \brief integrate dnnl softmax to softmax_output forward
 * \author Zhang Rong A
 */

#if MXNET_USE_ONEDNN == 1

#include "operator/softmax_output-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

static dnnl::softmax_forward::primitive_desc GetSoftmaxOutputFwdDescImpl(
    const SoftmaxOutputParam& param,
    bool is_train,
    const int axis,
    const dnnl::memory& input_mem) {
  dnnl::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine            = CpuEngine::Get()->get_engine();
  auto prop = is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_scoring;
  auto desc = dnnl::softmax_forward::desc(prop, data_md, axis);
  return dnnl::softmax_forward::primitive_desc(desc, cpu_engine);
}

typedef ParamOpSign<SoftmaxOutputParam> DNNLSoftmaxOuputSignature;

class DNNLSoftmaxOutputFwd {
  std::shared_ptr<dnnl::softmax_forward> fwd_;

 public:
  const dnnl::softmax_forward::primitive_desc fwd_pd;

  DNNLSoftmaxOutputFwd(const SoftmaxOutputParam& param,
                       bool is_train,
                       const int axis,
                       const dnnl::memory& mem)
      : fwd_pd(GetSoftmaxOutputFwdDescImpl(param, is_train, axis, mem)) {
    fwd_ = std::make_shared<dnnl::softmax_forward>(fwd_pd);
  }

  const inline dnnl::softmax_forward& GetFwd() const {
    return *fwd_;
  }
};

static DNNLSoftmaxOutputFwd& GetSoftmaxOutputForward(const SoftmaxOutputParam& param,
                                                     const OpContext& ctx,
                                                     const NDArray& in_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSoftmaxOuputSignature, DNNLSoftmaxOutputFwd, OpHash>
      fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSoftmaxOuputSignature, DNNLSoftmaxOutputFwd, OpHash>
      fwds;
#endif
  DNNLSoftmaxOuputSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(in_data);

  //  softmax_output has no axis parameter, so use it as it original implement.
  int axis = in_data.shape().ndim() - 1;

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto in_mem = *(in_data.GetDNNLData());
    DNNLSoftmaxOutputFwd fwd(param, ctx.is_train, axis, in_mem);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_softmax.html
bool SupportDNNLSoftmaxOutput(const SoftmaxOutputParam& param, const NDArray& input) {
  return SupportDNNL(input) && !param.multi_output;
}

void DNNLSoftmaxOutputForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& in_data,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& out_data) {
  const SoftmaxOutputParam& param = nnvm::get<SoftmaxOutputParam>(attrs.parsed);

  NDArray idata = in_data[softmaxout_enum::kData];
  NDArray odata = out_data[softmaxout_enum::kOut];
  if (in_data[softmaxout_enum::kData].IsView() && in_data[softmaxout_enum::kData].IsDNNLData()) {
    idata = in_data[softmaxout_enum::kData].Reorder2Default();
  }

  auto input_mem = idata.GetDNNLData();
  auto out_mem   = CreateDNNLMem(
      out_data[softmaxout_enum::kOut], input_mem->get_desc(), req[softmaxout_enum::kOut]);

  DNNLSoftmaxOutputFwd& fwd = GetSoftmaxOutputForward(param, ctx, idata);

  DNNLStream* stream = DNNLStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(),
                           {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DST, *out_mem.second}});
  CommitOutput(out_data[softmaxout_enum::kOut], out_mem);
  stream->Submit();
}
}  // namespace op
}  // namespace mxnet
#endif
