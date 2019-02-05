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
 * \file mkldnn_quantize_v2-inl.h
 * \brief
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_V2_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_V2_INL_H_
#if MXNET_USE_MKLDNN == 1
#include <algorithm>
#include <string>
#include <vector>
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../quantize_v2-inl.h"

namespace mxnet {
namespace op {

template <typename SrcType, typename DstType>
static void MKLDNNQuantizeComputeKer(const std::vector<NDArray>& inputs,
                                     const std::vector<NDArray>& outputs,
                                     const QuantizeV2Param& param,
                                     const std::vector<OpReqType>& req) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  SrcType real_range = 0.f;
  DstType quantized_range = 0;
  NDArray in_buffer = inputs[0];
  SrcType data_min = red::limits::MaxValue<SrcType>();
  SrcType data_max = red::limits::MinValue<SrcType>();
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    data_min = param.min_calib_range.value();
    data_max = param.max_calib_range.value();
  } else {
    // no calib info
    in_buffer = inputs[0].Reorder2Default();
    auto in_ptr = in_buffer.data().dptr<SrcType>();
    auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    std::vector<SrcType> data_maxs(nthreads, data_max);
    std::vector<SrcType> data_mins(nthreads, data_min);
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(in_buffer.shape().Size()); i++) {
      int tid = omp_get_thread_num();
      if (in_ptr[i] > data_maxs[tid]) data_maxs[tid] = in_ptr[i];
      if (in_ptr[i] < data_mins[tid]) data_mins[tid] = in_ptr[i];
    }
    for (index_t i = 0; i < nthreads; i++) {
      if (data_maxs[i] > data_max) data_max = data_maxs[i];
      if (data_mins[i] < data_min) data_min = data_mins[i];
    }
  }

  auto out_type = GetOutputType(param);
  if (out_type == mshadow::kUint8) {
    real_range = std::max<SrcType>(0.f, data_max);
    quantized_range = MaxValue<DstType>();
    *outputs[1].data().dptr<float>() = 0.f;
    *outputs[2].data().dptr<float>() = real_range;
  } else if (out_type == mshadow::kInt8) {
    real_range = MaxAbs(data_min, data_max);
    quantized_range = MinAbs(MaxValue<DstType>(), MinValue<DstType>());
    *outputs[1].data().dptr<float>() = -real_range;
    *outputs[2].data().dptr<float>() = real_range;
  } else {
    LOG(FATAL) << "mkldnn quantize op only supports int8 and uint8 as output type";
  }
  float scale = static_cast<float>(quantized_range) / real_range;

  primitive_attr attr;
  const int mask = 0;
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  attr.set_int_output_round_mode(round_nearest);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();

  if (in_buffer.IsView() && in_buffer.IsMKLDNNData()) in_buffer = inputs[0].Reorder2Default();
  auto i_mem = in_buffer.GetMKLDNNData();
  auto i_mpd = i_mem->get_primitive_desc();
  auto i_desc = i_mpd.desc();
  mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(i_desc.data.format);
  if (i_fmt == mkldnn::memory::format::nchw ||
      i_fmt == mkldnn::memory::format::nChw8c ||
      i_fmt == mkldnn_nChw16c) {
    i_fmt = mkldnn::memory::format::nhwc;
  }
  size_t i_ndim = in_buffer.shape().ndim();
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
  for (size_t i = 0; i < i_ndim; i++) {
    i_dims[i] = static_cast<int>(in_buffer.shape()[i]);
  }
  auto o_desc =
      mkldnn::memory::desc(i_dims, (mkldnn::memory::data_type)data_type_enum<DstType>::type, i_fmt);
  auto o_mpd = memory::primitive_desc(o_desc, cpu_engine);
  auto reorder_pd = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto o_mem = CreateMKLDNNMem(outputs[0], o_mpd, req[0]);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *i_mem, *o_mem.second));
  CommitOutput(outputs[0], o_mem);
  MKLDNNStream::Get()->Submit();
}

static void MKLDNNQuantizeV2Compute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  const QuantizeV2Param& param = nnvm::get<QuantizeV2Param>(attrs.parsed);
  auto out_type = GetOutputType(param);
  if (out_type == mshadow::kUint8) {
    MKLDNNQuantizeComputeKer<float, uint8_t>(inputs, outputs, param, req);
  } else if (out_type == mshadow::kInt8) {
    MKLDNNQuantizeComputeKer<float, int8_t>(inputs, outputs, param, req);
  } else {
    LOG(FATAL) << "mkldnn quantize op only supports int8 and uint8 as output type";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_V2_INL_H_
