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
 * \file quantized_concat.cc
 * \brief
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_concat-inl.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

namespace quantized_concat_enum {
enum QuantizedConcatOutputs { kOut, kMin, kMax };
}

static float GetScale(const NDArray& data, float min, float max) {
  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  return data_range / MaxAbs(min, max);
}

static void DNNLQuantizedConcatForward(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& in_data,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& out_data) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_data.size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_data.size(), 3U);
  // Collect data min/max and output_neg_min, output_pos_max
  std::vector<float> data_min(param_.num_args);
  std::vector<float> data_max(param_.num_args);
  float output_neg_min = 0.f;  // 0.f is the maximum for output_neg_min
  float output_pos_max = 0.f;  // 0.f is the minimum for output_pos_max
  for (int i = 0; i < param_.num_args; ++i) {
    data_min[i] = in_data[param_.num_args + 2 * i].data().dptr<float>()[0];
    if (data_min[i] < output_neg_min)
      output_neg_min = data_min[i];
    data_max[i] = in_data[param_.num_args + 2 * i + 1].data().dptr<float>()[0];
    if (data_max[i] > output_pos_max)
      output_pos_max = data_max[i];
  }
  out_data[quantized_concat_enum::kMin].data().dptr<float>()[0] = output_neg_min;
  out_data[quantized_concat_enum::kMax].data().dptr<float>()[0] = output_pos_max;
  auto out_scale = GetScale(out_data[quantized_concat_enum::kOut], output_neg_min, output_pos_max);
  std::vector<dnnl::memory::desc> data_md;
  std::vector<const dnnl::memory*> data_mem;
  // new_data_mem is for auto-free new created dnnl memory
  std::vector<std::shared_ptr<dnnl::memory>> new_data_mem;
  const auto out_dtype = out_data[quantized_concat_enum::kOut].dtype();
  for (int i = 0; i < param_.num_args; ++i) {
    auto i_scale = GetScale(in_data[i], data_min[i], data_max[i]);
    if (i_scale == out_scale) {
      CHECK(in_data[i].dtype() == out_dtype);
      auto mem = in_data[i].GetDNNLData();
      data_mem.push_back(mem);
      data_md.push_back(mem->get_desc());
    } else {
      auto mem      = in_data[i].GetDNNLData();
      auto mem_desc = mem->get_desc();
      if (in_data[i].dtype() != out_dtype) {
        mem_desc.data.data_type = static_cast<dnnl_data_type_t>(get_dnnl_type(out_dtype));
      }
      const auto rescaled_mem =
          std::make_shared<dnnl::memory>(mem_desc, CpuEngine::Get()->get_engine());
      new_data_mem.push_back(rescaled_mem);
      std::vector<float> reorder_scale = {out_scale / i_scale};
      dnnl::primitive_attr reorder_attr;
      reorder_attr.set_output_scales(0, reorder_scale);
      const auto reorder_pd = dnnl::reorder::primitive_desc(*mem, *rescaled_mem, reorder_attr);
      dnnl_args_map_t reorder_args;
      reorder_args[DNNL_ARG_SRC] = *mem;
      reorder_args[DNNL_ARG_DST] = *rescaled_mem;
      DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd), reorder_args);
      data_mem.push_back(rescaled_mem.get());
      data_md.push_back(mem_desc);
    }
  }
  int param_dim                = param_.dim.has_value() ? param_.dim.value() : 0;
  param_dim                    = CheckAxis(param_dim, in_data[concat_enum::kData0].shape().ndim());
  DNNLConcatFwd& fwd           = DNNLConcatFwd::GetCached(param_dim, in_data, data_md);
  mxnet::dnnl_output_t out_mem = CreateDNNLMem(
      out_data[quantized_concat_enum::kOut], fwd.fwd_pd.dst_desc(), req[concat_enum::kOut]);
  dnnl_args_map_t net_args;
  net_args[DNNL_ARG_DST] = *out_mem.second;
  for (int i = 0; i < param_.num_args; i++) {
    net_args[DNNL_ARG_MULTIPLE_SRC + i] = *data_mem[i];
  }
  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  DNNLStream::Get()->Submit();
}

inline static bool ConcatStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_attrs->size(), 3U);

  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_concat)
    .set_attr<FInferStorageType>("FInferStorageType", ConcatStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedConcatForward)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
