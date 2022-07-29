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
 * \file dnnl_binary.cc
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#if MXNET_USE_ONEDNN == 1
#include "./dnnl_binary-inl.h"

namespace mxnet {
namespace op {

DNNLBinaryOpFwd::DNNLBinaryOpFwd(const dnnl::algorithm alg,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<NDArray>& outputs) {
  auto src0_desc = inputs[0].GetDNNLData()->get_desc();
  auto src1_desc = inputs[1].GetDNNLData()->get_desc();
  auto dst_desc  = outputs[0].GetDNNLData()->get_desc();

  dnnl::binary::desc fwd_desc(alg, src0_desc, src1_desc, dst_desc);
  fwd_pd = std::make_shared<binary_fwd_pd_t>(fwd_desc, mxnet::CpuEngine::Get()->get_engine());
  fwd    = std::make_shared<binary_fwd_t>(*fwd_pd);
}

void DNNLBinaryOpFwd::Execute(const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  auto src0   = inputs[0].GetDNNLData();
  auto src1   = inputs[1].GetDNNLData();
  dnnl_output_t out_mem;
  if (outputs[0].GetDNNLData()->get_data_handle() == inputs[1].GetDNNLData()->get_data_handle())
    out_mem = CreateDNNLMem(outputs[0], fwd_pd->dst_desc(), req[0], &inputs[1]);
  else
    out_mem = CreateDNNLMem(outputs[0], fwd_pd->dst_desc(), req[0], &inputs[0]);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC_0, *src0},
      {DNNL_ARG_SRC_1, *src1},
      {DNNL_ARG_DST, *out_mem.second},
  };

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[0], out_mem);
  DNNLStream::Get()->Submit();
}

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_binary.html
bool SupportDNNLBinary(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs) {
  // threshold value selected experimentally basing on performance results - PR-21106
  constexpr size_t optimal_size_threshold = 2 << 13;
  const bool threshold_condition          = outputs[0].shape().Size() >= optimal_size_threshold;
  const bool is_any_dnnl_data =
      inputs[0].IsDNNLData() || inputs[1].IsDNNLData() || outputs[0].IsDNNLData();

  return SupportDNNL<DNNLTypeMode::FloatTypes>(inputs[0]) &&
         SupportDNNL<DNNLTypeMode::FloatTypes>(inputs[1]) &&
         (threshold_condition || is_any_dnnl_data);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
