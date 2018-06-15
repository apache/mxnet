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
 * \file mkldnn_concat.cc
 * \brief
 * \author Wenting Jiang
*/
#include "../concat-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

class MKLDNNConcatFwd {
  std::shared_ptr<mkldnn::concat> fwd;
  std::vector<std::shared_ptr<mkldnn::memory>> data;
  std::vector<mkldnn::primitive::at> data_mem;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::concat::primitive_desc fwd_pd;

  MKLDNNConcatFwd(
      int concat_dim,
      const std::vector<mkldnn::memory::primitive_desc> &data_md): fwd_pd(concat_dim, data_md) {
    data.resize(data_md.size());
  }

  void SetNewMem(const std::vector<const mkldnn::memory *> &in_data,
                 const mkldnn::memory &output) {
    CHECK_EQ(in_data.size(), data.size());
    for (size_t i = 0; i < data.size(); i++) {
      if (this->data[i] == nullptr) {
        this->data[i] = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                in_data[i]->get_primitive_desc(), in_data[i]->get_data_handle()));
        this->data_mem.push_back(*this->data[i]);
      } else {
        this->data[i]->set_data_handle(in_data[i]->get_data_handle());
      }
    }
    if (this->out == nullptr)
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output.get_data_handle()));
    else
      this->out->set_data_handle(output.get_data_handle());

    if (this->fwd == nullptr)
      fwd.reset(new mkldnn::concat(fwd_pd, data_mem, *out));
  }

  const mkldnn::concat &GetFwd() const {
    return *fwd;
  }
};

static MKLDNNConcatFwd &GetConcatForward(
    int concat_dim, const std::vector<NDArray> &in_data,
    const std::vector<mkldnn::memory::primitive_desc> &data_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, MKLDNNConcatFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, MKLDNNConcatFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(concat_dim);
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConcatFwd fwd(concat_dim, data_md);
    auto ins_ret = fwds.insert(std::pair<OpSignature, MKLDNNConcatFwd>(
            key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNConcatForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                         const std::vector<NDArray> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_in_data = param.num_args;
  int concat_dim = param.dim;
  std::vector<mkldnn::memory::primitive_desc> data_md;
  std::vector<const mkldnn::memory *> data_mem;
  data_md.reserve(num_in_data);
  data_mem.reserve(num_in_data);
  for (int i =0; i < num_in_data; i++) {
    const mkldnn::memory *tmp_mem = in_data[i].GetMKLDNNData();
    mkldnn::memory::primitive_desc tmp_pd = tmp_mem->get_primitive_desc();
    data_md.push_back(tmp_pd);
    data_mem.push_back(tmp_mem);
  }
  MKLDNNConcatFwd &fwd = GetConcatForward(concat_dim, in_data, data_md);
  mxnet::mkldnn_output_t out_mem = CreateMKLDNNMem(out_data[concat_enum::kOut],
                                                   fwd.fwd_pd.dst_primitive_desc(),
                                                   req[concat_enum::kOut]);
  fwd.SetNewMem(data_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

void MKLDNNConcatBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_in_data = param.num_args;
  int axis_ = param.dim;
  auto engine = CpuEngine::Get()->get_engine();
  auto gz_mem = inputs[0].GetMKLDNNData();
  mkldnn::memory::primitive_desc gz_pd = gz_mem->get_primitive_desc();
  /* init the offset */
  mkldnn::memory::dims offsets = {0, 0, 0, 0};
  for (int i = 0; i < num_in_data; i++) {
    mkldnn::memory::dims diff_src_tz
        = {static_cast<int>(inputs[i+1].shape()[0]),
          static_cast<int>(inputs[i+1].shape()[1]),
          static_cast<int>(inputs[i+1].shape()[2]),
          static_cast<int>(inputs[i+1].shape()[3])};
    auto diff_src_mpd = inputs[i+1].GetMKLDNNData()->get_primitive_desc();
    auto gradi_mem_ = CreateMKLDNNMem(outputs[i], diff_src_mpd, req[i]);
    // create view from gy to gxs[i]
    std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
    view_pd.reset(new mkldnn::view::primitive_desc(gz_pd, diff_src_tz, offsets));
    // create reorder primitive from gy to gxs[i]
    mkldnn::reorder::primitive_desc reorder_pd(
        view_pd.get()->dst_primitive_desc(), diff_src_mpd);
    offsets[axis_] += diff_src_tz[axis_];
    MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(
            reorder_pd, *gz_mem, *gradi_mem_.second));
    CommitOutput(outputs[i], gradi_mem_);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
