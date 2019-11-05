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
 * \file mkldnn_quantize-inl.h
 * \brief
 * \author Wenting Jiang, Xinyu Chen
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
#if MXNET_USE_MKLDNN == 1
#include <string>
#include <algorithm>
#include <vector>
#include "../quantize-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

template <typename SrcType, typename DstType>
static void MKLDNNQuantizeComputeKer(const std::vector<NDArray>& inputs,
                                     const std::vector<NDArray>& outputs,
                                     const QuantizeParam& param,
                                     const std::vector<OpReqType>& req) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  float real_range = 0.0;
  float quantized_range = 0.0;
  if (param.out_type == mshadow::kUint8) {
    real_range = MaxAbs(*inputs[1].data().dptr<float>(), *inputs[2].data().dptr<float>());
    quantized_range = MaxAbs(MaxValue<DstType>(), MinValue<DstType>());
    *outputs[1].data().dptr<float>() = *inputs[1].data().dptr<float>();
    *outputs[2].data().dptr<float>() = *inputs[2].data().dptr<float>();
  } else if (param.out_type == mshadow::kInt8) {
    real_range = MaxAbs(*inputs[1].data().dptr<float>(), *inputs[2].data().dptr<float>());
    quantized_range = MinAbs(MaxValue<DstType>(), MinValue<DstType>());
    *outputs[1].data().dptr<float>() = -real_range;
    *outputs[2].data().dptr<float>() = real_range;
  } else {
    LOG(FATAL) << "mkldnn quantize op only supports int8 and uint8 as output type";
  }
  float scale = quantized_range / real_range;
  mkldnn::primitive_attr attr;
  const int mask = 0;
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  NDArray in_buffer = inputs[0];
  if (inputs[0].IsView() && inputs[0].IsMKLDNNData()) in_buffer = inputs[0].Reorder2Default();

  auto i_mem = in_buffer.GetMKLDNNData();
  auto i_desc = i_mem->get_desc();
  size_t i_ndim = in_buffer.shape().ndim();
  mkldnn::memory::desc o_desc;
  if (i_ndim == 4) {
    mkldnn::memory::format_tag o_fmt = mkldnn::memory::format_tag::nhwc;
    mkldnn::memory::dims o_dims(i_desc.data.dims, i_desc.data.dims + i_desc.data.ndims);
    o_desc = mkldnn::memory::desc(o_dims, get_mkldnn_type<DstType>(), o_fmt);
  } else {
    o_desc = i_desc;
    o_desc.data.data_type = get_mkldnn_type_t<DstType>();
  }
  auto reorder_pd = mkldnn::reorder::primitive_desc(cpu_engine, i_desc, cpu_engine, o_desc, attr);
  auto o_mem = CreateMKLDNNMem(outputs[0], o_desc, req[0]);
  MKLDNNStream::Get()->RegisterPrimArgs(
      mkldnn::reorder(reorder_pd), {{MKLDNN_ARG_FROM, *i_mem}, {MKLDNN_ARG_TO, *o_mem.second}});
  CommitOutput(outputs[0], o_mem);
  MKLDNNStream::Get()->Submit();
}

static void MKLDNNQuantizeCompute(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  if (param.out_type == mshadow::kUint8) {
    MKLDNNQuantizeComputeKer<float, uint8_t>(inputs, outputs, param, req);
  } else if (param.out_type == mshadow::kInt8) {
    MKLDNNQuantizeComputeKer<float, int8_t>(inputs, outputs, param, req);
  } else {
    LOG(FATAL) << "mkldnn quantize op only supports int8 and uint8 as output type";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
