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
 * \file mkldnn_slice.cc
 * \brief
 * \author Zhiyuan Huang
*/

#if MXNET_USE_MKLDNN == 1

#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_slice-inl.h"

namespace mxnet {
namespace op {

MKLDNNSliceFwd::MKLDNNSliceFwd(const SliceParam &param,
                               const NDArray &in,
                               const NDArray &out) {
  const TShape ishape = in.shape();
  const TShape oshape = out.shape();
  uint32_t N = ishape.ndim();
  mkldnn::memory::dims dims(N);
  mkldnn::memory::dims offsets(N);
  for (uint32_t i = 0; i < N; ++i) {
    int s = 0;
    if (param.begin[i]) {
      s = *param.begin[i];
      if (s < 0) s += ishape[i];
    }
    dims[i] = oshape[i];
    offsets[i] = s;
  }
  auto in_mem_pd = in.GetMKLDNNData()->get_primitive_desc();
  auto out_mem_pd = out.GetMKLDNNData()->get_primitive_desc();
  auto view_pd = mkldnn::view::primitive_desc(in_mem_pd, dims, offsets);
  auto reorder_pd = reorder::primitive_desc(view_pd.dst_primitive_desc(), out_mem_pd);
  this->data_ = std::make_shared<mkldnn::memory>(view_pd.dst_primitive_desc(), nullptr);
  this->out_ = std::make_shared<mkldnn::memory>(view_pd.dst_primitive_desc(), nullptr);
  this->fwd_ = std::make_shared<mkldnn::reorder>(reorder_pd, *this->data_, *this->out_);
}

void MKLDNNSliceFwd::SetNewMem(const mkldnn::memory &input, const mkldnn::memory &output) {
  this->data_->set_data_handle(input.get_data_handle());
  this->out_->set_data_handle(output.get_data_handle());
}

const mkldnn::reorder &MKLDNNSliceFwd::GetPd() const {
  return *fwd_;
}

MKLDNNSliceFwd &GetSliceForward(const SliceParam &param, const bool is_train,
                                const NDArray &in_data, const NDArray &out_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNSliceSignature, MKLDNNSliceFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNSliceSignature, MKLDNNSliceFwd, OpHash> fwds;
#endif
  MKLDNNSliceSignature key(param);
  key.AddSign(is_train);
  key.AddSign(in_data);
  key.AddSign(out_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSliceFwd fwd(param, in_data, out_data);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNSlice(const SliceParam &param, const OpContext& ctx,
                 const NDArray &in, OpReqType req, const NDArray &out) {
  MKLDNNSliceFwd &fwd = GetSliceForward(param, ctx.is_train, in, out);
  auto in_mem = in.GetMKLDNNData();
  auto out_mem_pd = out.GetMKLDNNData()->get_primitive_desc();
  auto out_mem = CreateMKLDNNMem(out, out_mem_pd, req);
  fwd.SetNewMem(*in_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetPd());
  CommitOutput(out, out_mem);
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
