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

void MKLDNNSum(const std::vector<const mkldnn::memory*> arrs,
               const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(arrs.size());
  std::vector<float> scales(arrs.size(), 1);
  std::vector<mkldnn::primitive::at> inputs;

  mkldnn::memory::primitive_desc prev_pd;
  mkldnn::memory::primitive_desc tmp_pd;
  for (size_t i = 0; i < arrs.size(); i++) {
    input_pds[i] = arrs[i]->get_primitive_desc();
    inputs.push_back(*arrs[i]);
  }


  mkldnn::sum::primitive_desc sum_pd(scales, input_pds);
  // check if inplace sum is possible
  auto in_place = false;
  for (size_t i = 0; i < arrs.size(); i++) {
    if (input_pds[i] == sum_pd.dst_primitive_desc() && arrs[i]->get_data_handle() == out.get_data_handle())
      in_place = true;
  }
  if (in_place) {
    // do sum computation directly on output NDArray
    MKLDNNStream::Get()->RegisterPrim(mkldnn::sum(sum_pd, inputs, out));
  } else {
    auto sum_res = TmpMemMgr::Get()->Alloc(out.get_primitive_desc());
    MKLDNNStream::Get()->RegisterPrim(mkldnn::sum(sum_pd, inputs, *sum_res));
    CopyMKLDNNMem(*sum_res, &out);
  }
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
