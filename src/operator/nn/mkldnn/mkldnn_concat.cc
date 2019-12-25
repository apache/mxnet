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
 * \author
*/

#if MXNET_USE_MKLDNN == 1
#include "mkldnn_concat-inl.h"

namespace mxnet {
namespace op {

const mkldnn::concat &MKLDNNConcatFwd::GetFwd() const { return *fwd_; }

void MKLDNNConcatForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                         const std::vector<NDArray> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  const int num_in_data = param.num_args;
  const int concat_dim = param.dim;
  std::vector<mkldnn::memory::desc> data_md;
  std::vector<const mkldnn::memory *> data_mem;
  data_md.reserve(num_in_data);
  data_mem.reserve(num_in_data);
  for (int i = 0; i < num_in_data; i++) {
    const mkldnn::memory *tmp_mem = in_data[i].GetMKLDNNData();
    mkldnn::memory::desc tmp_md = tmp_mem->get_desc();
    data_md.push_back(tmp_md);
    data_mem.push_back(tmp_mem);
  }
  MKLDNNConcatFwd &fwd = GetConcatForward(concat_dim, in_data, data_md);
  mxnet::mkldnn_output_t out_mem = CreateMKLDNNMem(out_data[concat_enum::kOut],
                                                   fwd.fwd_pd.dst_desc(),
                                                   req[concat_enum::kOut]);
  std::unordered_map<int, mkldnn::memory> net_args;
  net_args.insert({MKLDNN_ARG_DST, *out_mem.second});
  for (int i = 0; i < num_in_data; i++) {
    net_args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, *data_mem[i]});
  }
  MKLDNNStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

void MKLDNNConcatBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  const int num_in_data = param.num_args;
  const int axis = param.dim;
  const auto gradz_mem = inputs[0].GetMKLDNNData();
  /* init the offset */
  mkldnn::memory::dims offsets(outputs[0].shape().ndim());
  for (auto &v : offsets) {
    v = 0;
  }

  for (int i = 0; i < num_in_data; i++) {
    mkldnn::memory::dims diff_src_tz(outputs[i].shape().begin(), outputs[i].shape().end());
    auto diff_src_md = outputs[i].GetMKLDNNData()->get_desc();
    auto gradi_mem = CreateMKLDNNMem(outputs[i], diff_src_md, req[i]);

    auto from_md = gradz_mem->get_desc().submemory_desc(diff_src_tz, offsets);
    auto from_mem = new mkldnn::memory(from_md, gradz_mem->get_engine(),
                                       gradz_mem->get_data_handle());
    offsets[axis] += diff_src_tz[axis];

    std::unordered_map<int, mkldnn::memory> net_args({
        {MKLDNN_ARG_FROM, *gradz_mem},
        {MKLDNN_ARG_TO, *gradi_mem.second}
    });
    MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::reorder(*from_mem, *gradi_mem.second), net_args);
    CommitOutput(outputs[i], gradi_mem);
  }

  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
