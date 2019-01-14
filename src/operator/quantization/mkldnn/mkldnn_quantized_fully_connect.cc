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
 * Copyright (c) 2018 by Contributors
 * \file quantized_fully_connected.cc
 * \brief
 */

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

namespace quantized_fully_connected_enum {
enum QuantizedFullyConnectedOutputs { kOut, kMin, kMax };
}


static void MKLDNNQuantizedFullyConnectedForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                                                 const std::vector<NDArray>& in_data,
                                                 const std::vector<OpReqType>& req,
                                                 const std::vector<NDArray>& out_data) {
  const FullyConnected & param_ = nnvm::get<FullyConnectedParam>(attrs.parsed);
  CHECK_EQ(in_data.size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_data.size(), 3U);

  NDArray weight = in_data[fullc::kWeight]
  auto &fwd = GetFCFwd();

  // Collect data min/max and output_neg_min, output_pos_max
  std::vector<float> data_min(param_.num_args);
  std::vector<float> data_max(param_.num_args);
  float output_neg_min = 0.f;  // 0.f is the maximum for output_neg_min
  float output_pos_max = 0.f;  // 0.f is the minimum for output_pos_max
  for (int i = 0; i < param_.num_args; ++i) {
    data_min[i] = in_data[param_.num_args + 2 * i].data().dptr<float>()[0];
    if (data_min[i] < output_neg_min) output_neg_min = data_min[i];
    data_max[i] = in_data[param_.num_args + 2 * i + 1].data().dptr<float>()[0];
    if (data_max[i] > output_pos_max) output_pos_max = data_max[i];
  }
  out_data[quantized_concat_enum::kMin].data().dptr<float>()[0] = output_neg_min;
  out_data[quantized_concat_enum::kMax].data().dptr<float>()[0] = output_pos_max;
  auto out_scale = GetScale(out_data[quantized_concat_enum::kOut], output_neg_min, output_pos_max);
  std::vector<mkldnn::memory::primitive_desc> data_md;
  std::vector<const mkldnn::memory*> data_mem;
  // new_data_mem is for auto-free new created mkldnn memory
  std::vector<std::shared_ptr<mkldnn::memory>> new_data_mem;
  for (int i = 0; i < param_.num_args; ++i) {
    auto i_scale = GetScale(in_data[i], data_min[i], data_max[i]);
    if (i_scale == out_scale) {
      auto mem = in_data[i].GetMKLDNNData();
      data_mem.push_back(mem);
      data_md.push_back(mem->get_primitive_desc());
    } else {
      auto mem = in_data[i].GetMKLDNNData();
      auto pd = mem->get_primitive_desc();
      const auto rescaled_mem = std::make_shared<mkldnn::memory>(pd);
      new_data_mem.push_back(rescaled_mem);
      std::vector<float> reorder_scale = {out_scale / i_scale};
      primitive_attr reorder_attr;
      reorder_attr.set_int_output_round_mode(round_mode::round_nearest);
      reorder_attr.set_output_scales(0, reorder_scale);
      const auto reorder_pd = mkldnn::reorder::primitive_desc(pd, pd, reorder_attr);
      MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *mem, *rescaled_mem));
      data_mem.push_back(rescaled_mem.get());
      data_md.push_back(pd);
    }
  }
  MKLDNNConcatFwd& fwd = GetConcatForward(param_.dim, in_data, data_md);
  mxnet::mkldnn_output_t out_mem =
      CreateMKLDNNMem(out_data[quantized_concat_enum::kOut], fwd.fwd_pd.dst_primitive_desc(),
                      req[concat_enum::kOut]);
  fwd.SetNewMem(data_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

NNVM_REGISTER_OP(_contrib_fully_connected)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedFullyConnectedForward)
.set_attr<bool>("TIsMKLDNN", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
