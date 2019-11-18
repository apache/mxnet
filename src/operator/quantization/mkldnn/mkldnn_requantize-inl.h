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

/* \file mkldnn_requantize-inl.h
 * \brief
 * \author Jin Huang, Xinyu Chen
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_REQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_REQUANTIZE_INL_H_
#if MXNET_USE_MKLDNN == 1
#include <string>
#include <algorithm>
#include <vector>
#include "../requantize-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

template <typename DstType>
static void MKLDNNRequantizeForwardKer(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs,
                                       const float real_range) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  typedef int32_t SrcDType;
  // check shapes
  size_t i_dim = inputs[0].shape().ndim();
  size_t o_dim = outputs[0].shape().ndim();
  CHECK_EQ(i_dim, o_dim);
  float first_quantized_range = MinAbs(MinValue<SrcDType>(),
                                       MaxValue<SrcDType>());
  float first_real_range = MaxAbs(*inputs[1].data().dptr<float>(),
                                  *inputs[2].data().dptr<float>());
  float first_scale = first_real_range / first_quantized_range;
  float second_real_range = real_range;
  float second_quantized_range = 0.f;
  if (std::is_same<DstType, int8_t>::value) {
    second_quantized_range = MinAbs(MaxValue<DstType>(), MinValue<DstType>());
    *outputs[1].data().dptr<float>() = -second_real_range;
    *outputs[2].data().dptr<float>() = second_real_range;
  } else if (std::is_same<DstType, uint8_t>::value) {
    second_quantized_range = MaxValue<DstType>();
    *outputs[1].data().dptr<float>() = 0.f;
    *outputs[2].data().dptr<float>() = second_real_range;
  } else {
    LOG(FATAL) << "Unsupported requantize output type";
  }
  float second_scale = second_quantized_range / second_real_range;
  float scale = first_scale * second_scale;

  mkldnn::primitive_attr attr;
  const int mask = 0;
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();

  NDArray in_buffer = inputs[0];
  if (inputs[0].IsView() && inputs[0].IsMKLDNNData())
    in_buffer = inputs[0].Reorder2Default();

  auto i_mem = in_buffer.GetMKLDNNData();
  auto i_desc = i_mem->get_desc();
  auto o_desc = i_desc;
  o_desc.data.data_type = get_mkldnn_type_t<DstType>();
  auto reorder_pd  = mkldnn::reorder::primitive_desc(cpu_engine, i_desc, cpu_engine, o_desc, attr);
  auto o_mem = CreateMKLDNNMem(outputs[0], o_desc, req[0]);
  MKLDNNStream::Get()->RegisterPrimArgs(
      mkldnn::reorder(reorder_pd), {{MKLDNN_ARG_FROM, *i_mem}, {MKLDNN_ARG_TO, *o_mem.second}});
  CommitOutput(outputs[0], o_mem);
  MKLDNNStream::Get()->Submit();
}

static void MKLDNNRequantizeForward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  const RequantizeParam& param = nnvm::get<RequantizeParam>(attrs.parsed);
  float real_range;
  // Model is calibrated
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    real_range =
          MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
  // Model is not calibrated
  } else {
    NDArray in_buffer = inputs[0].Reorder2Default();
    auto in_ptr = in_buffer.data().dptr<SrcDType>();
    auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    SrcDType data_min = MaxValue<SrcDType>();
    SrcDType data_max = MinValue<SrcDType>();
    std::vector<SrcDType> data_maxs(nthreads, data_max);
    std::vector<SrcDType> data_mins(nthreads, data_min);
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
    float src_range = MinAbs(MinValue<SrcDType>(), MaxValue<SrcDType>());
    SrcDType data_range = MaxAbs(data_min, data_max);
    float data_scale = MaxAbs(*inputs[1].data().dptr<float>(), *inputs[2].data().dptr<float>());
    real_range = data_range * data_scale / src_range;
  }
  auto out_type = GetQuantizeOutputType(param);
  if (out_type == mshadow::kUint8) {
    MKLDNNRequantizeForwardKer<uint8_t>(attrs, ctx, inputs, req, outputs, real_range);
  } else if (out_type == mshadow::kInt8) {
    MKLDNNRequantizeForwardKer<int8_t>(attrs, ctx, inputs, req, outputs, real_range);
  } else {
    LOG(FATAL) << "mkldnn requantize op only supports int8 and uint8 as output type";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_REQUANTIZE_INL_H_
