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
 * \file mkldnn_dot.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "./mkldnn_dot-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNDot(const std::vector<NDArray>& inputs, const NDArray& output) {
  return inputs[0].shape().Size() != 0 && inputs[1].shape().Size() != 0 &&
         output.shape().Size() != 0 && output.shape().ndim() <= 6 &&
         (inputs[0].dtype() == mshadow::kFloat32 || inputs[0].dtype() == mshadow::kBfloat16);
}

void MKLDNNDotForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                      const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  MKLDNNDotFwd& fwd     = MKLDNNDotFwd::GetCached(param, inputs, outputs);
  fwd.Execute(inputs, req, outputs);
}

MKLDNNDotFwd& MKLDNNDotFwd::GetCached(const DotParam& param, const std::vector<NDArray>& inputs,
                                      const std::vector<NDArray>& outputs) {
  using dot_fwd_map = std::unordered_map<DotSignature, MKLDNNDotFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local dot_fwd_map fwds;
#else
  static MX_THREAD_LOCAL dot_fwd_map fwds;
#endif

  DotSignature key(param);
  key.AddSign(inputs[0]);
  key.AddSign(inputs[1]);
  key.AddSign(outputs[0]);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const MKLDNNDotFwd fwd(param, inputs, outputs);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

MKLDNNDotFwd::MKLDNNDotFwd(const DotParam& param, const std::vector<NDArray>& inputs,
                           const std::vector<NDArray>& outputs) {
  auto shapeData = inputs[0].shape(), shapeWeights = inputs[1].shape();
  auto ndimData = shapeData.ndim(), ndimWeights = shapeWeights.ndim();
  int smallDimData, bigDimData, smallDimWeights, bigDimWeights;

  smallDimData    = param.transpose_a ? shapeData[0] : shapeData[ndimData - 1];
  bigDimData      = shapeData.Size() / smallDimData;
  smallDimWeights = param.transpose_b ? shapeWeights[ndimWeights - 1] : shapeWeights[0];
  bigDimWeights   = shapeWeights.Size() / smallDimWeights;

  auto GetMemoryDesc = [](const NDArray& tensor, int firstDim, int secondDim,
                          const bool transpose) {
    return mkldnn::memory::desc(
        mkldnn::memory::dims{firstDim, secondDim}, get_mkldnn_type(tensor.dtype()),
        transpose ? mkldnn::memory::format_tag::ba : mkldnn::memory::format_tag::any);
  };

  mkldnn::memory::desc data_md =
      GetMemoryDesc(inputs[0], bigDimData, smallDimData, param.transpose_a);
  mkldnn::memory::desc weights_md =
      GetMemoryDesc(inputs[1], smallDimWeights, bigDimWeights, param.transpose_b);
  mkldnn::memory::desc out_md({bigDimData, bigDimWeights}, get_mkldnn_type(outputs[0].dtype()),
                              mkldnn::memory::format_tag::any);
  mkldnn::matmul::desc fwd_desc(data_md, weights_md, out_md);
  fwd_pd = std::make_shared<dot_fwd_pd_t>(fwd_desc, mxnet::CpuEngine::Get()->get_engine());
  fwd    = std::make_shared<dot_fwd_t>(*fwd_pd);
}

void MKLDNNDotFwd::Execute(const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  auto data =
      mkldnn::memory(fwd_pd->src_desc(), engine, reinterpret_cast<void*>(inputs[0].data().dptr_));
  auto weights            = mkldnn::memory(fwd_pd->weights_desc(), engine,
                                reinterpret_cast<void*>(inputs[1].data().dptr_));
  mkldnn_output_t out_mem = CreateMKLDNNMem(outputs[0], fwd_pd->dst_desc(), req[0], &inputs[0]);

  mkldnn_args_map_t args = {
      {MKLDNN_ARG_SRC, data},
      {MKLDNN_ARG_WEIGHTS, weights},
      {MKLDNN_ARG_DST, *out_mem.second},
  };

  MKLDNNStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[0], out_mem);
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
