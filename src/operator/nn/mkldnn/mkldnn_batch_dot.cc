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
 * \file mkldnn_batch_dot.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "./mkldnn_batch_dot-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNBatchDot(const std::vector<NDArray> &inputs,
                           const NDArray &output) {
  return inputs[0].shape().Size() != 0 && inputs[1].shape().Size() != 0 &&
         output.shape().Size() != 0 &&
         (inputs[0].dtype() == mshadow::kFloat32 ||
          inputs[0].dtype() == mshadow::kBfloat16);
}

void MKLDNNBatchDotForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  const DotParam &param = nnvm::get<DotParam>(attrs.parsed);
  MKLDNNBatchDotFwd &fwd = MKLDNNBatchDotFwd::GetCached(param, inputs, outputs);
  fwd.Execute(inputs, req, outputs);
}

MKLDNNBatchDotFwd &MKLDNNBatchDotFwd::GetCached(
    const DotParam &param, const std::vector<NDArray> &inputs,
    const std::vector<NDArray> &outputs) {
  using batch_dot_fwd_map =
      std::unordered_map<BatchDotSignature, MKLDNNBatchDotFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local batch_dot_fwd_map fwds;
#else
  static MX_THREAD_LOCAL batch_dot_fwd_map fwds;
#endif

  BatchDotSignature key(param);
  key.AddSign(inputs[0]);
  key.AddSign(inputs[1]);
  key.AddSign(outputs[0]);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const MKLDNNBatchDotFwd fwd(param, inputs, outputs);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

MKLDNNBatchDotFwd::MKLDNNBatchDotFwd(const DotParam &param,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<NDArray> &outputs) {
  auto shape = inputs[0].shape();
  auto ndim = shape.ndim();
  auto bigDim = shape[0];
  for (size_t i = 1; i < ndim - 2; ++i) {
    bigDim *= shape[i];
  }

  auto GetMemoryDesc = [&ndim, &bigDim](const NDArray &tensor,
                                        const bool transpose) {
    auto shape = tensor.shape();
    if (transpose) {
      return mkldnn::memory::desc(
          mkldnn::memory::dims{bigDim, shape[ndim - 1], shape[ndim - 2]},
          get_mkldnn_type(tensor.dtype()), mkldnn::memory::format_tag::acb);
    } else {
      return mkldnn::memory::desc(
          mkldnn::memory::dims{bigDim, shape[ndim - 2], shape[ndim - 1]},
          get_mkldnn_type(tensor.dtype()), mkldnn::memory::format_tag::any);
    }
  };

  mkldnn::memory::desc data_md = GetMemoryDesc(inputs[0], param.transpose_a);
  mkldnn::memory::desc weights_md = GetMemoryDesc(inputs[1], param.transpose_b);
  mkldnn::memory::desc out_md({bigDim, data_md.dims()[1], weights_md.dims()[2]},
                              get_mkldnn_type(outputs[0].dtype()),
                              mkldnn::memory::format_tag::any);
  mkldnn::matmul::desc fwd_desc(data_md, weights_md, out_md);
  fwd_pd = std::make_shared<batch_dot_fwd_pd_t>(
      fwd_desc, mxnet::CpuEngine::Get()->get_engine());
  fwd = std::make_shared<batch_dot_fwd_t>(*fwd_pd);
}

void MKLDNNBatchDotFwd::Execute(const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  auto data = mkldnn::memory(fwd_pd->src_desc(), engine,
                             reinterpret_cast<void *>(inputs[0].data().dptr_));
  auto weights =
      mkldnn::memory(fwd_pd->weights_desc(), engine,
                     reinterpret_cast<void *>(inputs[1].data().dptr_));
  mkldnn_output_t out_mem =
      CreateMKLDNNMem(outputs[0], fwd_pd->dst_desc(), req[0], &inputs[0]);

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
