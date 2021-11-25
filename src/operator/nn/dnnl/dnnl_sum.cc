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

#include "operator/operator_common.h"
#include "dnnl_base-inl.h"
#include "dnnl_ops-inl.h"

namespace mxnet {
namespace op {

#if MXNET_USE_ONEDNN == 1
void DNNLSum(const dnnl::memory& arr1, const dnnl::memory& arr2, const dnnl::memory& out) {
  std::vector<dnnl::memory::desc> input_pds(2);
  std::vector<float> scales(2, 1);
  input_pds[0] = arr1.get_desc();
  input_pds[1] = arr2.get_desc();
  CHECK(input_pds[0] == input_pds[0]);
  const dnnl::memory* in_mem1 = &arr1;
  const dnnl::memory* in_mem2 = &arr2;
  auto output_pd              = out.get_desc();
  if (input_pds[0] != output_pd) {
    auto tmp_memory1 = TmpMemMgr::Get()->Alloc(output_pd);
    auto tmp_memory2 = TmpMemMgr::Get()->Alloc(output_pd);
    DNNLMemoryCopy(arr1, tmp_memory1);
    DNNLMemoryCopy(arr2, tmp_memory2);
    input_pds[0] = tmp_memory1->get_desc();
    input_pds[1] = tmp_memory2->get_desc();
    in_mem1      = tmp_memory1;
    in_mem2      = tmp_memory2;
  }
  dnnl::sum::primitive_desc sum_pd(output_pd, scales, input_pds, CpuEngine::Get()->get_engine());
  dnnl_args_map_t args = {
      {DNNL_ARG_MULTIPLE_SRC, *in_mem1},
      {DNNL_ARG_MULTIPLE_SRC + 1, *in_mem2},
      {DNNL_ARG_DST, out},
  };
  DNNLStream::Get()->RegisterPrimArgs(dnnl::sum(sum_pd), args);
}

class DNNLSumFwd {
 public:
  dnnl::sum::primitive_desc fwd_pd;

  DNNLSumFwd(const std::vector<float>& scales, const std::vector<dnnl::memory::desc>& data_md)
      : fwd_pd(scales, data_md, CpuEngine::Get()->get_engine()) {
    fwd_ = std::make_shared<dnnl::sum>(fwd_pd);
  }

  const dnnl::sum& GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<dnnl::sum> fwd_;
};

static DNNLSumFwd& GetSumForward(const std::vector<float>& scales,
                                 const std::vector<NDArray>& in_data,
                                 const std::vector<dnnl::memory::desc>& data_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, DNNLSumFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, DNNLSumFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLSumFwd fwd(scales, data_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLSumForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  const int num_inputs    = inputs.size();
  const NDArray& out_data = outputs[0];
  std::vector<dnnl::memory::desc> data_md;
  std::vector<const dnnl::memory*> data_mem;
  std::vector<float> scales(num_inputs, 1);

  data_md.reserve(num_inputs);
  data_mem.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    const dnnl::memory* in_mem = inputs[i].GetDNNLData();
    dnnl::memory::desc tmp_md  = in_mem->get_desc();
    data_md.push_back(tmp_md);
    data_mem.push_back(in_mem);
  }

  DNNLSumFwd& fwd              = GetSumForward(scales, inputs, data_md);
  mxnet::dnnl_output_t out_mem = CreateDNNLMem(out_data, fwd.fwd_pd.dst_desc(), req[0], &inputs[0]);
  dnnl_args_map_t net_args;
  net_args.insert({DNNL_ARG_DST, *out_mem.second});
  for (int i = 0; i < num_inputs; ++i) {
    net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, *data_mem[i]});
  }
  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data, out_mem);
  DNNLStream::Get()->Submit();
}
#endif

}  // namespace op
}  // namespace mxnet
