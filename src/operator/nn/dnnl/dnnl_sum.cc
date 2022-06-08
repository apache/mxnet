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
 * \file dnnl_sum.cc
 * \brief
 * \author Da Zheng
 */
#include <iostream>
#include <vector>

#include "operator/operator_common.h"
#include "dnnl_base-inl.h"
#include "dnnl_sum-inl.h"

namespace mxnet {
namespace op {

#if MXNET_USE_ONEDNN == 1

DNNLSumFwd& DNNLSumFwd::GetCached(const std::vector<NDArray>& inputs,
                                  const std::vector<NDArray>& outputs) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSumSignature, DNNLSumFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSumSignature, DNNLSumFwd, OpHash> fwds;
#endif
  DNNLSumSignature key;
  key.AddSign(inputs);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLSumFwd fwd(inputs, outputs);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

DNNLSumFwd::DNNLSumFwd(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs) {
  const int num_inputs    = inputs.size();

  std::vector<dnnl::memory::desc> data_md;

  std::vector<float> scales(num_inputs, 1);

  data_md.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    const dnnl::memory* in_mem = inputs[i].GetDNNLData();
    dnnl::memory::desc tmp_md  = in_mem->get_desc();
    data_md.push_back(tmp_md);
  }

  fwd_pd = std::make_shared<sum_pd_t>(scales, data_md, CpuEngine::Get()->get_engine());
  fwd    = std::make_shared<sum_t>(*fwd_pd);
}

void DNNLSumFwd::Execute(const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  const NDArray& out_data = outputs[0];
  const int num_inputs    = inputs.size();
  std::vector<const dnnl::memory*> data_mem;

  data_mem.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    const dnnl::memory* in_mem = inputs[i].GetDNNLData();
    data_mem.push_back(in_mem);
  }

  mxnet::dnnl_output_t out_mem = CreateDNNLMem(out_data, fwd_pd->dst_desc(), req[0], &inputs[0]);
  dnnl_args_map_t net_args;
  net_args.insert({DNNL_ARG_DST, *out_mem.second});
  for (int i = 0; i < num_inputs; ++i) {
    net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, *data_mem[i]});
  }
  DNNLStream::Get()->RegisterPrimArgs(*fwd, net_args);
  CommitOutput(out_data, out_mem);
  DNNLStream::Get()->Submit();
}

void DNNLSumForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  DNNLSumFwd& fwd = DNNLSumFwd::GetCached(inputs, outputs);
  fwd.Execute(ctx, inputs, req, outputs);
}
#endif

}  // namespace op
}  // namespace mxnet
