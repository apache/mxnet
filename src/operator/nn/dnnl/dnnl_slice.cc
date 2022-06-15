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
 * \file dnnl_slice.cc
 * \brief
 * \author Zhiyuan Huang
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_base-inl.h"
#include "dnnl_slice-inl.h"

namespace mxnet {
namespace op {

DNNLSliceFwd::DNNLSliceFwd(const SliceParam& param, const NDArray& in, const NDArray& out) {
  const mxnet::TShape ishape = in.shape();
  const mxnet::TShape oshape = out.shape();
  const int N                = ishape.ndim();
  dnnl::memory::dims dims(N);
  dnnl::memory::dims offsets(N);
  for (int i = 0; i < N; ++i) {
    dim_t s = 0;
    if (i < param.begin.ndim() && param.begin[i]) {
      s = *param.begin[i];
      if (s < 0)
        s += ishape[i];
    }
    dims[i]    = oshape[i];
    offsets[i] = s;
  }

  auto in_md  = in.GetDNNLData()->get_desc();
  auto out_md = out.GetDNNLData()->get_desc();
  auto sub_md = in_md.submemory_desc(dims, offsets);

  auto engine = CpuEngine::Get()->get_engine();
  this->data_ = std::make_shared<dnnl::memory>(sub_md, engine, nullptr);
  this->out_  = std::make_shared<dnnl::memory>(out_md, engine, nullptr);
  this->fwd_  = std::make_shared<dnnl::reorder>(*this->data_, *this->out_);
}

void DNNLSliceFwd::SetNewMem(const dnnl::memory& input, const dnnl::memory& output) {
  this->data_->set_data_handle(input.get_data_handle());
  this->out_->set_data_handle(output.get_data_handle());
}

void DNNLSliceFwd::Register() {
  DNNLStream::Get()->RegisterPrimArgs(
      *fwd_, {{DNNL_ARG_FROM, *(this->data_)}, {DNNL_ARG_TO, *(this->out_)}});
}

DNNLSliceFwd& GetSliceForward(const SliceParam& param,
                              const bool is_train,
                              const NDArray& in_data,
                              const NDArray& out_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSliceSignature, DNNLSliceFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSliceSignature, DNNLSliceFwd, OpHash> fwds;
#endif
  DNNLSliceSignature key(param);
  key.AddSign(is_train);
  key.AddSign(in_data);
  key.AddSign(out_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLSliceFwd fwd(param, in_data, out_data);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLSlice(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const NDArray& in,
               OpReqType req,
               const NDArray& out) {
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  DNNLSliceFwd& fwd       = GetSliceForward(param, ctx.is_train, in, out);
  auto in_mem             = in.GetDNNLData();
  auto out_md             = out.GetDNNLData()->get_desc();
  auto out_mem            = CreateDNNLMem(out, out_md, req);
  fwd.SetNewMem(*in_mem, *out_mem.second);
  fwd.Register();
  CommitOutput(out, out_mem);
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
