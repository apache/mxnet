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
 * \file mkldnn_softmax_output.cc
 * \brief integrate mkldnn softmax to softmax_output forward
 * \author Zhang Rong A
*/

#if MXNET_USE_MKLDNN == 1
#include "../../softmax_output-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
namespace mxnet {
namespace op {

static mkldnn::softmax_forward::primitive_desc GetSoftmaxOutputFwdDescImpl(
               const SoftmaxOutputParam& param, bool is_train,
               const int axis, const mkldnn::memory &input_mem) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto prop = is_train ? mkldnn::prop_kind::forward_training
                       : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::softmax_forward::desc(prop, data_md, axis);
  return mkldnn::softmax_forward::primitive_desc(desc, cpu_engine);
}

typedef ParamOpSign<SoftmaxOutputParam> MKLDNNSoftmaxOuputSignature;

class MKLDNNSoftmaxOutputFwd {
  std::shared_ptr<mkldnn::softmax_forward> fwd_;

 public:
  const mkldnn::softmax_forward::primitive_desc fwd_pd;

  MKLDNNSoftmaxOutputFwd(const SoftmaxOutputParam& param, bool is_train,
                         const int axis, const mkldnn::memory &mem): fwd_pd(
                         GetSoftmaxOutputFwdDescImpl(param, is_train, axis, mem)) {
    fwd_ = std::make_shared<mkldnn::softmax_forward>(fwd_pd);
  }

  const inline mkldnn::softmax_forward &GetFwd() const {
    return *fwd_;
  }
};

static MKLDNNSoftmaxOutputFwd &GetSoftmaxOutputForward(const SoftmaxOutputParam& param,
                                                       const OpContext &ctx,
                                                       const NDArray &in_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
    std::unordered_map<MKLDNNSoftmaxOuputSignature, MKLDNNSoftmaxOutputFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL
    std::unordered_map<MKLDNNSoftmaxOuputSignature, MKLDNNSoftmaxOutputFwd, OpHash> fwds;
#endif
  MKLDNNSoftmaxOuputSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(in_data);

  //  softmax_output has no axis parameter, so use it as it original implement.
  int axis = in_data.shape().ndim() - 1;

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto in_mem = *(in_data.GetMKLDNNData());
    MKLDNNSoftmaxOutputFwd fwd(param, ctx.is_train, axis, in_mem);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

//  This is only used for forward. For backward ,need double check compatibility
bool SupportMKLDNNSoftmaxOutput(const SoftmaxOutputParam &param) {
  return param.multi_output ? false : true;
}

void MKLDNNSoftmaxOutputForward(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  const SoftmaxOutputParam &param = nnvm::get<SoftmaxOutputParam>(attrs.parsed);

  NDArray idata = in_data[softmaxout_enum::kData];
  NDArray odata = out_data[softmaxout_enum::kOut];
  if (in_data[softmaxout_enum::kData].IsView() && in_data[softmaxout_enum::kData].IsMKLDNNData()) {
    idata = in_data[softmaxout_enum::kData].Reorder2Default();
  }

  auto input_mem = idata.GetMKLDNNData();
  auto out_mem = CreateMKLDNNMem(out_data[softmaxout_enum::kOut],
                                 input_mem->get_desc(), req[softmaxout_enum::kOut]);

  MKLDNNSoftmaxOutputFwd &fwd = GetSoftmaxOutputForward(param, ctx, idata);

  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(),
                           {{MKLDNN_ARG_SRC, *input_mem}, {MKLDNN_ARG_DST, *out_mem.second}});
  CommitOutput(out_data[softmaxout_enum::kOut], out_mem);
  stream->Submit();
}
}   // namespace op
}   // namespace mxnet
#endif

