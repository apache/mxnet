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
 * \file mkldnn_quantized_elemwise_add.cc
 * \brief
 */

#if MXNET_USE_MKLDNN == 1
#include "../quantized_elemwise_add-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuantizeElemwiseAddParam);

static inline float GetScale(const NDArray& data, float min, float max) {
  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  return data_range / MaxAbs(min, max);
}

class MKLDNNQuantizedElemwiseAddFwd {
 public:
  mkldnn::sum::primitive_desc fwd_pd;

  MKLDNNQuantizedElemwiseAddFwd(
               const mkldnn::memory::desc &output_desc,
               const std::vector<float> &scales,
               const std::vector<mkldnn::memory::desc> &data_md)
      : fwd_pd(output_desc, scales, data_md, CpuEngine::Get()->get_engine()) {
    fwd_ = std::make_shared<mkldnn::sum>(fwd_pd);
    data_.resize(data_md.size());
  }

  const mkldnn::sum &GetFwd() const { return *fwd_; }

 private:
  std::shared_ptr<mkldnn::sum> fwd_;
  std::vector<std::shared_ptr<mkldnn::memory>> data_;
  std::shared_ptr<mkldnn::memory> out_;
};

static MKLDNNQuantizedElemwiseAddFwd &GetQuantizedElemwiseAddForward(
    const mkldnn::memory::desc &output_desc, const std::vector<float> &scales,
    const std::vector<NDArray> &in_data, const std::vector<NDArray> &out_data,
    const std::vector<mkldnn::memory::desc> &data_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature,
              MKLDNNQuantizedElemwiseAddFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature,
              MKLDNNQuantizedElemwiseAddFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(in_data);
  key.AddSign(in_data[quantized_elemwise_add_enum::kAMin].data().dptr<float>()[0]);
  key.AddSign(in_data[quantized_elemwise_add_enum::kAMax].data().dptr<float>()[0]);
  key.AddSign(in_data[quantized_elemwise_add_enum::kBMin].data().dptr<float>()[0]);
  key.AddSign(in_data[quantized_elemwise_add_enum::kBMax].data().dptr<float>()[0]);
  key.AddSign(out_data);
  key.AddSign(out_data[quantized_elemwise_add_enum::kMin].data().dptr<float>()[0]);
  key.AddSign(out_data[quantized_elemwise_add_enum::kMax].data().dptr<float>()[0]);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNQuantizedElemwiseAddFwd fwd(output_desc, scales, data_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}


static void MKLDNNQuantizedElemwiseAddForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                                              const std::vector<NDArray>& in_data,
                                              const std::vector<OpReqType>& req,
                                              const std::vector<NDArray>& out_data) {
  const QuantizeElemwiseAddParam& params = nnvm::get<QuantizeElemwiseAddParam>(attrs.parsed);
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_data.size(), 6U) << "should be A, B, A_min, A_max, B_min, B_max";
  // C, C_min, C_max
  CHECK_EQ(out_data.size(), 3U) << "should be C, C_min, C_max";
  // Collect data min,max,absmax
  const float dataA_min = in_data[quantized_elemwise_add_enum::kAMin].data().dptr<float>()[0];
  const float dataB_min = in_data[quantized_elemwise_add_enum::kBMin].data().dptr<float>()[0];
  const float dataA_max = in_data[quantized_elemwise_add_enum::kAMax].data().dptr<float>()[0];
  const float dataB_max = in_data[quantized_elemwise_add_enum::kBMax].data().dptr<float>()[0];
  const float dataA_absmax = MaxAbs(dataA_min, dataA_max);
  const float dataB_absmax = MaxAbs(dataB_min, dataB_max);

  auto dataA_mem  = in_data[quantized_elemwise_add_enum::kDataA].GetMKLDNNData();
  auto dataB_mem  = in_data[quantized_elemwise_add_enum::kDataB].GetMKLDNNData();
  const bool is_dataA_int8 = (in_data[quantized_elemwise_add_enum::kDataA].dtype()
                              == mshadow::kInt8);
  const float dataA_range = is_dataA_int8 ? kInt8Range : kUint8Range;

  const float A_scale = GetScale(in_data[quantized_elemwise_add_enum::kDataA],
                                 dataA_min,
                                 dataA_max);
  const float B_scale = GetScale(in_data[quantized_elemwise_add_enum::kDataB],
                                 dataB_min,
                                 dataB_max);
  // rescaled_mem is for reorder mkldnn memory
  mkldnn::memory *rescaled_mem;

  // output default set as int32
  float output_data_range = kInt32Range;
  auto output_data_type = mkldnn::memory::data_type::s32;
  // dataA && dataB are uint8
  if (out_data[quantized_elemwise_add_enum::kOut].dtype() == mshadow::kInt8) {
    output_data_range = kInt8Range;
    output_data_type = mkldnn::memory::data_type::s8;
  } else if (out_data[quantized_elemwise_add_enum::kOut].dtype() == mshadow::kUint8) {
    output_data_range = kUint8Range;
    output_data_type = mkldnn::memory::data_type::u8;
  } else {
    output_data_range = kInt32Range;
    output_data_type = mkldnn::memory::data_type::s32;
  }

  float output_min = 0;
  float output_max = 0;
  float out_data_scale = 0;
  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    output_min = params.min_calib_range.value();
    output_max = params.max_calib_range.value();
    out_data_scale = output_data_range / MaxAbs(output_min, output_max);
  } else {
    output_max = dataA_absmax + dataB_absmax;
    output_min = -output_max;
  }
  // 2: scale 0 for dataA, scale 1 for data B
  const int scales_num = 2;
  std::vector<float> scales(scales_num, 1);
  auto engine = CpuEngine::Get()->get_engine();
  if (in_data[quantized_elemwise_add_enum::kDataA].dtype()
      != in_data[quantized_elemwise_add_enum::kDataB].dtype()) {
    auto s8_desc = (is_dataA_int8 == true)
                 ? dataA_mem->get_desc()
                 : dataB_mem->get_desc();
    rescaled_mem = TmpMemMgr::Get()->Alloc(s8_desc);
    float u8_reorder_scale = 0;
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      if (is_dataA_int8 == true) {
        u8_reorder_scale = out_data_scale / B_scale;
        scales[0] = out_data_scale / A_scale;
      } else {
        u8_reorder_scale = out_data_scale / A_scale;
        scales[1] = out_data_scale / B_scale;
      }
    } else {
      // x*dataA_absmax/dataA_range = y*(dataA_absmax+dataB_absmax)/output_range
      if (is_dataA_int8 == true) {
        u8_reorder_scale = dataB_absmax * output_data_range
                           / ((dataA_absmax + dataB_absmax) * kUint8Range);
        scales[0] = dataA_absmax * output_data_range
                         / ((dataA_absmax + dataB_absmax) * dataA_range);
      } else {
        u8_reorder_scale = dataA_absmax * output_data_range
                           / ((dataA_absmax + dataB_absmax) * dataA_range);
        scales[1] = dataB_absmax * output_data_range
                         / ((dataA_absmax + dataB_absmax) * kInt8Range);
      }
    }
    std::vector<float> reorder_scale = {u8_reorder_scale};
    mkldnn::primitive_attr reorder_attr;
    reorder_attr.set_output_scales(0, reorder_scale);
    auto u8_mem = (is_dataA_int8 == true) ? dataB_mem : dataA_mem;
    const auto reorder_pd = mkldnn::reorder::primitive_desc(engine,
                                                            u8_mem->get_desc(),
                                                            engine,
                                                            s8_desc,
                                                            reorder_attr);
    mkldnn_args_map_t args({{MKLDNN_ARG_FROM, *u8_mem }, {MKLDNN_ARG_TO, *rescaled_mem}});
    MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::reorder(reorder_pd), args);

    if (is_dataA_int8 == true) {
      dataB_mem = rescaled_mem;
    } else {
      dataA_mem = rescaled_mem;
    }
  } else {
    // same data type and has same data range
    if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
      scales[0] = out_data_scale / A_scale;
      scales[1] = out_data_scale / B_scale;
    } else {
      scales[0] = dataA_absmax * output_data_range / ((dataA_absmax + dataB_absmax) * dataA_range);
      scales[1] = dataB_absmax * output_data_range / ((dataA_absmax + dataB_absmax) * dataA_range);
    }
  }

  std::vector<mkldnn::memory::desc> in_desc;
  in_desc.push_back(dataA_mem->get_desc());
  in_desc.push_back(dataB_mem->get_desc());
  const auto in_shape = in_data[quantized_elemwise_add_enum::kDataA].shape();
  mkldnn::memory::dims i_dims(in_shape.begin(), in_shape.end());
  auto output_desc = mkldnn::memory::desc(i_dims,
                                          output_data_type,
                                          mkldnn::memory::format_tag::any);
  MKLDNNQuantizedElemwiseAddFwd &fwd = GetQuantizedElemwiseAddForward(output_desc, scales,
                                                                      in_data, out_data, in_desc);
  auto mem = CreateMKLDNNMem(out_data[quantized_elemwise_add_enum::kOut],
                             fwd.fwd_pd.dst_desc(),
                             req[0],
                             &in_data[0]);
  mkldnn_args_map_t args({{MKLDNN_ARG_MULTIPLE_SRC,  *dataA_mem},
                          {MKLDNN_ARG_MULTIPLE_SRC + 1,  *dataB_mem},
                          {MKLDNN_ARG_DST, *mem.second}});
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), args);
  CommitOutput(out_data[quantized_elemwise_add_enum::kOut], mem);
  stream->Submit();

  out_data[quantized_elemwise_add_enum::kMin].data().dptr<float>()[0] = output_min;
  out_data[quantized_elemwise_add_enum::kMax].data().dptr<float>()[0] = output_max;
}

inline static bool ElemwiseAddStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                          DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  // Check num of inputs: A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_attrs->size(), 6U);
  // Check num of outputs: C, C_min, C_max
  CHECK_EQ(out_attrs->size(), 3U);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_elemwise_add)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedElemwiseAddForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr_parser(ParamParser<QuantizeElemwiseAddParam>)
.add_arguments(QuantizeElemwiseAddParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
