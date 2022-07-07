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
 * \file dnnl_pow_mul_scalar.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_pow_mul_scalar-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLPowMulScalarParam);

DNNLPowMulScalarFwd& DNNLPowMulScalarFwd::GetCached(const DNNLPowMulScalarParam& param,
                                                    const NDArray& input,
                                                    const NDArray& output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLPowMulScalarSignature, DNNLPowMulScalarFwd, OpHash>
      fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLPowMulScalarSignature, DNNLPowMulScalarFwd, OpHash>
      fwds;
#endif
  DNNLPowMulScalarSignature key(param);
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLPowMulScalarFwd fwd(param, input);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

DNNLPowMulScalarFwd::DNNLPowMulScalarFwd(const DNNLPowMulScalarParam& param, const NDArray& input) {
  auto src_desc = input.GetDNNLData()->get_desc();
  dnnl::eltwise_forward::desc fwd_desc(dnnl::prop_kind::forward_scoring,
                                       dnnl::algorithm::eltwise_pow,
                                       src_desc,
                                       param.multiplier,
                                       param.exponent);
  fwd_pd = std::make_shared<eltwise_fwd_pd_t>(fwd_desc, mxnet::CpuEngine::Get()->get_engine());
  fwd    = std::make_shared<eltwise_fwd_t>(*fwd_pd);
}

void DNNLPowMulScalarFwd::Execute(const NDArray& input,
                                  const OpReqType& req,
                                  const NDArray& output) {
  auto src              = input.GetDNNLData();
  dnnl_output_t out_mem = CreateDNNLMem(output, fwd_pd->dst_desc(), req, &input);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, *src},
      {DNNL_ARG_DST, *out_mem.second},
  };

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(output, out_mem);
  DNNLStream::Get()->Submit();
}
}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::DNNLPowMulScalarParam> {
  size_t operator()(const mxnet::op::DNNLPowMulScalarParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.exponent);
    ret        = dmlc::HashCombine(ret, val.multiplier);
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_USE_ONEDNN == 1
