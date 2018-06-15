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
#include <vector>

#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void MKLDNNSum(std::vector<mkldnn::memory> &in_mems,
         const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(in_mems.size());
  std::vector<float> scales(in_mems.size(), 1);
  std::vector<mkldnn::primitive::at> inputs;
  for (int i = 0; i < in_mems.size(); i++) {
    input_pds[i] = in_mems[i].get_primitive_desc();
    inputs.push_back(in_mems[i]);
  }
  // TODO(zhengda) I need to reorder memory here.
  mkldnn::sum::primitive_desc sum_pd(scales, input_pds);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::sum(sum_pd, inputs, out));
}

void MKLDNNSumForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs, const OpReqType &req,
                      const NDArray &out_data) {
  if (req == kNullOp) {
    return;
  }

  TmpMemMgr::Get()->Init(ctx.requested[0]);
  std::vector<mkldnn::memory> in_prims;
  std::vector<mkldnn::memory::primitive_desc> in_pds(inputs.size());
  std::vector<float> scales(inputs.size(), 1);
  in_prims.reserve(inputs.size());
  std::vector<NDArray> in_bufs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    const mkldnn::memory *in_mem;
    if (inputs[i].IsMKLDNNData() && inputs[i].IsView()) {
      in_bufs[i] = inputs[i].Reorder2Default();
      in_mem = in_bufs[i].GetMKLDNNData();
    } else {
      in_mem = inputs[i].GetMKLDNNData();
    }
    in_prims.push_back(*in_mem);
    in_pds[i] = in_mem->get_primitive_desc();
  }
  mkldnn::sum::primitive_desc pdesc(scales, in_pds);
  auto mem = CreateMKLDNNMem(out_data, pdesc.dst_primitive_desc(), req, &inputs[0]);
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::sum(pdesc, in_prims, *mem.second));
  CommitOutput(out_data, mem);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
