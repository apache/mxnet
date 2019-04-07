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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_quantized_sum.cc
 * \brief
 */

#if MXNET_USE_MKLDNN == 1
#include "./mkldnn_quantized_sum-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(RequantizeSumParam);

static float GetScale(const NDArray& data, float min, float max) {
  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  return data_range / MaxAbs(min, max);
}

static void MKLDNNQuantizedSumForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                                         const std::vector<NDArray>& in_data,
                                         const std::vector<OpReqType>& req,
                                         const std::vector<NDArray>& out_data) {
  const RequantizeSumParam& params = nnvm::get<RequantizeSumParam>(attrs.parsed);
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_data.size(), static_cast<size_t>(6));
  // C, C_min, C_max
  CHECK_EQ(out_data.size(), 3U);

  // Collect data min,max,absmax
  float dataA_min = in_data[quantized_sum_enum::kAMin].data().dptr<float>()[0];
  float dataB_min = in_data[quantized_sum_enum::kBMin].data().dptr<float>()[0];
  float dataA_max = in_data[quantized_sum_enum::kAMax].data().dptr<float>()[0];
  float dataB_max = in_data[quantized_sum_enum::kBMax].data().dptr<float>()[0];
  float dataA_absmax = MaxAbs(dataA_min, dataA_max);
  float dataB_absmax = MaxAbs(dataB_min, dataB_max);

  auto dataA_mem  = in_data[quantized_sum_enum::kDataA].GetMKLDNNData();
  auto dataB_mem  = in_data[quantized_sum_enum::kDataB].GetMKLDNNData();
  bool dataA_int8 = (in_data[quantized_sum_enum::kDataA].dtype() == mshadow::kInt8) ? true : false;
  size_t dataA_range = dataA_int8 ? kInt8Range : kUint8Range;

  float A_scale = GetScale(in_data[quantized_sum_enum::kDataA], dataA_min, dataA_max);
  float B_scale = GetScale(in_data[quantized_sum_enum::kDataB], dataB_min, dataB_max);
  // rescaled_mem is for reorder mkldnn memory
  std::shared_ptr<mkldnn::memory> rescaled_mem;
  // output default set as int32
  size_t output_data_range = kInt32Range;
  auto output_data_type = mkldnn::memory::s32;
  // dataA && dataB are uint8
  if (out_data[quantized_sum_enum::kDataA].dtype() == mshadow::kInt8) {
  output_data_range = kInt8Range;
  output_data_type = mkldnn::memory::s8;
  } else if (out_data[quantized_sum_enum::kDataA].dtype() == mshadow::kUint8) {
    output_data_range = kUint8Range;
    output_data_type = mkldnn::memory::u8;
  }

  float output_min = 0;
  float output_max = 0;
  float out_data_scale = 0;
  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    output_min = params.min_calib_range.value();
    output_max = params.max_calib_range.value();
    out_data_scale = output_data_range/MaxAbs(output_min, output_max);
  } else {
    output_max = dataA_absmax + dataB_absmax;
    output_min = 0 - output_max;
  }

  std::vector<float> scales;
  if (in_data[quantized_sum_enum::kDataA].dtype() != in_data[quantized_sum_enum::kDataB].dtype()) {
    auto s8_pd = (dataA_int8 == true)
                 ? dataA_mem->get_primitive_desc()
                 : dataB_mem->get_primitive_desc();
    rescaled_mem = std::make_shared<mkldnn::memory>(s8_pd);
    float u8_reorder_scale = 0;
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      if (dataA_int8 == true) {
        u8_reorder_scale = out_data_scale/B_scale;
        scales.push_back(out_data_scale/A_scale);
        scales.push_back(1);
      } else {
        u8_reorder_scale = out_data_scale/A_scale;
        scales.push_back(1);
        scales.push_back(out_data_scale/B_scale);
      }
    } else {
      // x*dataA_absmax/dataA_range = y*(dataA_absmax+dataB_absmax)/output_range
      if (dataA_int8 == true) {
        u8_reorder_scale = dataB_absmax*output_data_range
                           /((dataA_absmax + dataB_absmax)*kUint8Range);
        scales.push_back(dataA_absmax*output_data_range
                         /((dataA_absmax + dataB_absmax)*dataA_range));
        scales.push_back(1);
      } else {
        u8_reorder_scale = dataA_absmax*output_data_range
                           /((dataA_absmax + dataB_absmax)*dataA_range);
        scales.push_back(1);
        scales.push_back(dataB_absmax*output_data_range
                         /((dataA_absmax + dataB_absmax)*kInt8Range));
      }
    }
    std::vector<float> reorder_scale = {u8_reorder_scale};
    primitive_attr reorder_attr;
    reorder_attr.set_int_output_round_mode(round_mode::round_nearest);
    reorder_attr.set_output_scales(0, reorder_scale);
    auto u8_mem = (dataA_int8 == true) ? dataB_mem : dataA_mem;
    const auto reorder_pd = mkldnn::reorder::primitive_desc(u8_mem->get_primitive_desc(),
                                                            s8_pd,
                                                            reorder_attr);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *u8_mem, *rescaled_mem));

    if (dataA_int8 == true) {
      dataB_mem = rescaled_mem.get();
    } else {
      dataA_mem = rescaled_mem.get();
    }
  } else {
    // same data type and has same data range
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      scales.push_back(out_data_scale/A_scale);
      scales.push_back(out_data_scale/B_scale);
    } else {
      scales.push_back(dataA_absmax*output_data_range/((dataA_absmax + dataB_absmax)*dataA_range));
      scales.push_back(dataB_absmax*output_data_range/((dataA_absmax + dataB_absmax)*dataA_range));
    }
  }

  std::vector<mkldnn::primitive::at> in_prims;
  std::vector<mkldnn::memory::primitive_desc> in_pds;
  in_prims.push_back(*dataA_mem);
  in_prims.push_back(*dataB_mem);
  in_pds.push_back(dataA_mem->get_primitive_desc());
  in_pds.push_back(dataB_mem->get_primitive_desc());
  size_t i_ndim = in_data[quantized_sum_enum::kDataA].shape().ndim();
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
  for (size_t i = 0; i < i_ndim; i++) {
    i_dims[i] = static_cast<int>(in_data[quantized_sum_enum::kDataA].shape()[i]);
  }
  mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(
                                   in_pds[quantized_sum_enum::kDataA].desc().data.format);
  auto output_desc = memory::desc(i_dims, output_data_type, i_fmt);
  mkldnn::sum::primitive_desc pdesc(output_desc, scales, in_pds);
  auto mem = CreateMKLDNNMem(out_data[quantized_sum_enum::kOut],
                             pdesc.dst_primitive_desc(),
                             req[0],
                             &in_data[0]);
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::sum(pdesc, in_prims, *mem.second));
  CommitOutput(out_data[quantized_sum_enum::kOut], mem);
  stream->Submit();

  out_data[quantized_sum_enum::kMin].data().dptr<float>()[0] = output_min;
  out_data[quantized_sum_enum::kMax].data().dptr<float>()[0] = output_max;
}

inline static bool SumStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                     DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_attrs->size(), 6U);
  // C, C_min, C_max
  CHECK_EQ(out_attrs->size(), 3U);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_sum)
.set_attr<FInferStorageType>("FInferStorageType", SumStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedSumForward)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true)
.set_attr_parser(ParamParser<RequantizeSumParam>)
.add_arguments(RequantizeSumParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
