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

#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void MKLDNNSum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
         const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(2);
  std::vector<float> scales(2, 1);
  std::vector<mkldnn::primitive::at> inputs;
  input_pds[0] = arr1.get_primitive_desc();
  input_pds[1] = arr2.get_primitive_desc();
  CHECK(input_pds[0] == input_pds[0]);
  const mkldnn::memory *in_mem1 = &arr1;
  const mkldnn::memory *in_mem2 = &arr2;
  auto output_pd = out.get_primitive_desc();
  if (input_pds[0] != output_pd) {
    auto tmp_memory1 = TmpMemMgr::Get()->Alloc(output_pd);
    auto tmp_memory2 = TmpMemMgr::Get()->Alloc(output_pd);
    mxnet::MKLDNNCopy(arr1, tmp_memory1);
    mxnet::MKLDNNCopy(arr2, tmp_memory2);
    input_pds[0] = tmp_memory1->get_primitive_desc();
    input_pds[1] = tmp_memory2->get_primitive_desc();
    in_mem1 = tmp_memory1;
    in_mem2 = tmp_memory2;
  }
  inputs.push_back(*in_mem1);
  inputs.push_back(*in_mem2);
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
  std::vector<mkldnn::primitive::at> in_prims;
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
