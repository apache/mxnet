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
 * \file mkldnn_sum.cc
 * \brief
 * \author Da Zheng
*/
#include <iostream>

#include "../../operator_common.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void MKLDNNSum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
         const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(2);
  std::vector<float> scales(2, 1);
  std::vector<mkldnn::primitive::at> inputs;
  input_pds[0] = arr1.get_primitive_desc();
  input_pds[1] = arr2.get_primitive_desc();
  CHECK(input_pds[0] == input_pds[0]);
  const mkldnn::memory *in_mem1 = &arr1;
  const mkldnn::memory *in_mem2 = &arr2;
  auto output_pd = out.get_primitive_desc();
  if (input_pds[0] != output_pd) {
    auto tmp_memory1 = TmpMemMgr::Get()->Alloc(output_pd);
    auto tmp_memory2 = TmpMemMgr::Get()->Alloc(output_pd);
    mxnet::MKLDNNCopy(arr1, tmp_memory1);
    mxnet::MKLDNNCopy(arr2, tmp_memory2);
    input_pds[0] = tmp_memory1->get_primitive_desc();
    input_pds[1] = tmp_memory2->get_primitive_desc();
    in_mem1 = tmp_memory1;
    in_mem2 = tmp_memory2;
  }
  inputs.push_back(*in_mem1);
  inputs.push_back(*in_mem2);
  mkldnn::sum::primitive_desc sum_pd(scales, input_pds);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::sum(sum_pd, inputs, out));
}

class MKLDNNSumFwd {
 public:
  mkldnn::sum::primitive_desc fwd_pd;

  MKLDNNSumFwd(const std::vector<float> &scales,
               const std::vector<mkldnn::memory::primitive_desc> &data_md)
      : fwd_pd(scales, data_md) {
    data_.resize(data_md.size());
  }

  void SetNewMem(const std::vector<const mkldnn::memory *> &in_data, const mkldnn::memory &output);

  const mkldnn::sum &GetFwd() const { return *fwd_; }

 private:
  std::shared_ptr<mkldnn::sum> fwd_;
  std::vector<std::shared_ptr<mkldnn::memory>> data_;
  std::vector<mkldnn::primitive::at> data_mem_;
  std::shared_ptr<mkldnn::memory> out_;
};

static MKLDNNSumFwd &GetSumForward(
    const std::vector<float> &scales, const std::vector<NDArray> &in_data,
    const std::vector<mkldnn::memory::primitive_desc> &data_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, MKLDNNSumFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, MKLDNNSumFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSumFwd fwd(scales, data_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNSumFwd::SetNewMem(const std::vector<const mkldnn::memory *> &in_data,
                             const mkldnn::memory &output) {
  auto num_inputs = data_.size();
  CHECK_EQ(in_data.size(), num_inputs);
  for (index_t i = 0; i < static_cast<index_t>(num_inputs); ++i) {
    if (this->data_[i] == nullptr) {
      this->data_[i] = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(in_data[i]->get_primitive_desc(), in_data[i]->get_data_handle()));
      this->data_mem_.push_back(*this->data_[i]);
    } else {
      this->data_[i]->set_data_handle(in_data[i]->get_data_handle());
    }
  }
  if (this->out_ == nullptr)
    this->out_ = std::shared_ptr<mkldnn::memory>(
        new mkldnn::memory(fwd_pd.dst_primitive_desc(), output.get_data_handle()));
  else
    this->out_->set_data_handle(output.get_data_handle());

  if (this->fwd_ == nullptr)
    this->fwd_.reset(new mkldnn::sum(fwd_pd, this->data_mem_, *this->out_));
}

void MKLDNNSumForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs, const OpReqType &req,
                      const NDArray &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  auto num_inputs = inputs.size();
  std::vector<mkldnn::memory::primitive_desc> data_md;
  std::vector<const mkldnn::memory *> data_mem;
  std::vector<float> scales(num_inputs, 1);
  std::vector<NDArray> in_bufs(num_inputs);

  data_md.reserve(num_inputs);
  data_mem.reserve(num_inputs);

  for (index_t i = 0; i < static_cast<index_t>(num_inputs); ++i) {
    const mkldnn::memory *in_mem;
    if (inputs[i].IsMKLDNNData() && inputs[i].IsView()) {
      in_bufs[i] = inputs[i].Reorder2Default();
      in_mem = in_bufs[i].GetMKLDNNData();
    } else {
      in_bufs[i] = inputs[i];
      in_mem = inputs[i].GetMKLDNNData();
    }
    mkldnn::memory::primitive_desc tmp_pd = in_mem->get_primitive_desc();
    data_md.push_back(tmp_pd);
    data_mem.push_back(in_mem);
  }

  MKLDNNSumFwd &fwd = GetSumForward(scales, in_bufs, data_md);
  mxnet::mkldnn_output_t out_mem = CreateMKLDNNMem(out_data,
                                                   fwd.fwd_pd.dst_primitive_desc(),
                                                   req,
                                                   &in_bufs[0]);
  fwd.SetNewMem(data_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
  CommitOutput(out_data, out_mem);
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
