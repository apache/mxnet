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
 * \file dnnl_concat.cc
 * \brief
 * \author
 */

#if MXNET_USE_ONEDNN == 1
#include "dnnl_concat-inl.h"

namespace mxnet {
namespace op {

static inline bool IsUsingPadding(const dnnl::memory::desc& dst_md) {
  // make sure a blocked format is used (at least one dimension is blocked)
  bool is_blocked_format =
      dst_md.data.format_kind == dnnl_blocked && dst_md.data.format_desc.blocking.inner_nblks > 0;
  return is_blocked_format &&
         !std::equal(
             dst_md.data.dims, dst_md.data.dims + dst_md.data.ndims, dst_md.data.padded_dims);
}

DNNLConcatFwd::DNNLConcatFwd(int concat_dim, const std::vector<dnnl::memory::desc>& data_md)
    : fwd_pd(concat_dim, data_md, CpuEngine::Get()->get_engine()) {
  // DNNL introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, DNNL operators
  // still reuse memory from memory planning, so here we need to select a
  // format that has the expected memory size requirements (a plain format)

  // When fwd_pd uses padding, impose a plain format
  const auto& dst_md = fwd_pd.dst_desc();
  if (IsUsingPadding(dst_md)) {
    auto plain_dst_tag = static_cast<dnnl::memory::format_tag>(GetDefaultFormat(dst_md.data.ndims));
    auto plain_dst_md  = dnnl::memory::desc(dst_md.dims(), dst_md.data_type(), plain_dst_tag);
    fwd_pd             = dnnl::concat::primitive_desc(
        plain_dst_md, concat_dim, data_md, CpuEngine::Get()->get_engine());
  }
  fwd_ = std::make_shared<dnnl::concat>(fwd_pd);
}

DNNLConcatFwd& DNNLConcatFwd::GetCached(int concat_dim,
                                        const std::vector<NDArray>& in_data,
                                        const std::vector<dnnl::memory::desc>& data_md,
                                        int stack_axis /*used only by stack op*/) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, DNNLConcatFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, DNNLConcatFwd, OpHash> fwds;
#endif

  OpSignature key;
  key.AddSign(concat_dim);
  key.AddSign(stack_axis);
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLConcatFwd fwd(concat_dim, data_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLConcatForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<NDArray>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  const int num_in_data    = param.num_args;
  int concat_dim           = param.dim.has_value() ? param.dim.value() : 0;
  concat_dim               = CheckAxis(concat_dim, in_data[concat_enum::kData0].shape().ndim());
  std::vector<dnnl::memory::desc> data_md;
  std::vector<const dnnl::memory*> data_mem;
  data_md.reserve(num_in_data);
  data_mem.reserve(num_in_data);
  for (int i = 0; i < num_in_data; i++) {
    const dnnl::memory* tmp_mem = in_data[i].GetDNNLData();
    dnnl::memory::desc tmp_md   = tmp_mem->get_desc();
    data_md.push_back(tmp_md);
    data_mem.push_back(tmp_mem);
  }
  DNNLConcatFwd& fwd = DNNLConcatFwd::GetCached(concat_dim, in_data, data_md);
  mxnet::dnnl_output_t out_mem =
      CreateDNNLMem(out_data[concat_enum::kOut], fwd.fwd_pd.dst_desc(), req[concat_enum::kOut]);
  std::unordered_map<int, dnnl::memory> net_args;
  net_args.insert({DNNL_ARG_DST, *out_mem.second});
  for (int i = 0; i < num_in_data; i++) {
    net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, *data_mem[i]});
  }
  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  DNNLStream::Get()->Submit();
}

void DNNLConcatBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  const int num_in_data    = param.num_args;
  int concat_dim           = param.dim.has_value() ? param.dim.value() : 0;
  concat_dim               = CheckAxis(concat_dim, outputs[concat_enum::kData0].shape().ndim());
  const auto gradz_mem     = inputs[0].GetDNNLData();
  /* init the offset */
  dnnl::memory::dims offsets(outputs[0].shape().ndim());
  for (auto& v : offsets) {
    v = 0;
  }

  for (int i = 0; i < num_in_data; i++) {
    dnnl::memory::dims diff_src_tz(outputs[i].shape().begin(), outputs[i].shape().end());
    auto diff_src_md = outputs[i].GetDNNLData()->get_desc();
    auto gradi_mem   = CreateDNNLMem(outputs[i], diff_src_md, req[i]);

    auto from_md = gradz_mem->get_desc().submemory_desc(diff_src_tz, offsets);
    auto from_mem =
        new dnnl::memory(from_md, gradz_mem->get_engine(), gradz_mem->get_data_handle());
    offsets[concat_dim] += diff_src_tz[concat_dim];

    std::unordered_map<int, dnnl::memory> net_args(
        {{DNNL_ARG_FROM, *gradz_mem}, {DNNL_ARG_TO, *gradi_mem.second}});
    DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(*from_mem, *gradi_mem.second), net_args);
    CommitOutput(outputs[i], gradi_mem);
  }

  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
