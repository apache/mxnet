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

#include "../../softmax_output-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

static mkldnn::softmax_forward::primitive_desc GetSoftmaxOutputFwdDescImpl(
               const SoftmaxOutputParam& param, bool is_train,
               const NDArray &data, const mkldnn::memory &input_mem) {
  mkldnn::memory::primitive_desc data_mpd = input_mem.get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  //  softmax_output has no axis parameter, so use it as it original implement.
  int axis = data.shape().ndim() - 1;
  mkldnn::softmax_forward::desc desc = is_train
      ? mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_training,
                                      data_md, axis)
      : mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_scoring,
                                      data_md, axis);
  return mkldnn::softmax_forward::primitive_desc(desc, cpu_engine);
}

typedef ParamOpSign<SoftmaxOutputParam> MKLDNNSoftmaxOuputSignature;

class MKLDNNSoftmaxOutputFwd {
  std::shared_ptr<mkldnn::softmax_forward> fwd_;
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;

 public:
  const mkldnn::softmax_forward::primitive_desc fwd_pd;

  MKLDNNSoftmaxOutputFwd(const SoftmaxOutputParam& param, bool is_train,
                         const NDArray &data, const mkldnn::memory &mem): fwd_pd(
                         GetSoftmaxOutputFwdDescImpl(param, is_train, data, mem)) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &output) {
    if (this->data_ == nullptr)
      this->data_ = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
        data.get_primitive_desc(), data.get_data_handle()));
    else
      this->data_->set_data_handle(data.get_data_handle());

    if (this->out_ == nullptr)
      this->out_ = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
        output.get_primitive_desc(), output.get_data_handle()));
    else
      this->out_->set_data_handle(output.get_data_handle());

    if (this->fwd_ == nullptr) {
      this->fwd_ = std::shared_ptr<mkldnn::softmax_forward>(
        new mkldnn::softmax_forward(fwd_pd, mkldnn::primitive::at(*this->data_),
        *this->out_));
    }
  }

  const mkldnn::softmax_forward &GetFwd() const {
    return *fwd_;
  }
};

static MKLDNNSoftmaxOutputFwd &GetSoftmaxOutputForward(const SoftmaxOutputParam& param,
                                                       const OpContext &ctx,
                                                       const NDArray &in_data,
                                                       const mkldnn::memory &in_mem) {
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

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSoftmaxOutputFwd fwd(param, ctx.is_train, in_data, in_mem);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

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
  auto output_mem = odata.GetMKLDNNData();

  MKLDNNSoftmaxOutputFwd &fwd = GetSoftmaxOutputForward(param, ctx, idata, *input_mem);
  fwd.SetNewMem(*input_mem, *output_mem);
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(fwd.GetFwd());
  stream->Submit();
}
}   // namespace op
}   // namespace mxnet
#endif
