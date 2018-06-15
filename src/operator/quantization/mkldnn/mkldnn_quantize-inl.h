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

template<typename SrcType, typename DstType>
static void MKLDNNQuantizeComputeKer(const std::vector<NDArray>& inputs,
                                     const std::vector<NDArray>& outputs,
                                     const QuantizeParam& param,
                                     const std::vector<OpReqType> &req) {
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
  primitive_attr attr;
  const int mask = 0;
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  attr.set_int_output_round_mode(round_nearest);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();

  NDArray in_buffer = inputs[0];
  if (inputs[0].IsView() && inputs[0].IsMKLDNNData())
    in_buffer = inputs[0].Reorder2Default();

  auto i_mem = in_buffer.GetMKLDNNData();
  auto i_mpd = i_mem->get_primitive_desc();
  auto i_desc = i_mpd.desc();
  mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(i_desc.data.format);
  size_t i_ndim = in_buffer.shape().ndim();
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
  for (size_t i = 0; i < i_ndim; i++) {
    i_dims[i] = static_cast<int>(in_buffer.shape()[i]);
  }
  auto o_desc = mkldnn::memory::desc(i_dims,
                                    (mkldnn::memory::data_type)data_type_enum<DstType>::type,
                                    i_fmt);
  auto o_mpd = memory::primitive_desc(o_desc, cpu_engine);
  auto reorder_pd  = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto o_mem = CreateMKLDNNMem(outputs[0], o_mpd, req[0]);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *i_mem, *o_mem.second));
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
