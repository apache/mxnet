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
 * \author Jin Huang
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

void MKLDNNRequantizeForwardKer(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs,
                                const float real_range) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  // check shapes
  size_t i_dim = inputs[0].ndim();
  size_t o_dim = outputs[0].ndim();
  CHECK_EQ(i_dim, o_dim);
  unsigned int total_len = 1;
  memory::dims tensor_shape;
  for (size_t i = 0; i < i_dim; ++i) {
    CHECK_EQ(inputs[0].size(i), outputs[0].size(i));
    total_len *= inputs[0].size(i);
  }
  tensor_shape.push_back(total_len);
  float first_quantized_range = MinAbs(MinValue<SrcDType>(),
                                       MaxValue<SrcDType>());
  float first_real_range = MaxAbs(*inputs[1].dptr<float>(),
                                  *inputs[2].dptr<float>());
  float first_scale = first_real_range / first_quantized_range;
  float second_real_range = real_range;
  float second_quantized_range = MinAbs(MaxValue<DstDType>(),
                                        MinValue<DstDType>());
  float second_scale = second_quantized_range / second_real_range;
  float scale = first_scale * second_scale;
  *outputs[1].dptr<float>() = -second_real_range;
  *outputs[2].dptr<float>() = second_real_range;
  primitive_attr attr;
  const int mask = 0;
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  attr.set_int_output_round_mode(round_nearest);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  auto i_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<SrcDType>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto o_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<DstDType>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto reorder_pd  = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto input = memory(i_mpd, inputs[0].dptr<SrcDType>());
  auto output = memory(o_mpd, outputs[0].dptr<DstDType>());
  auto r = reorder(reorder_pd, input, output);
  stream(stream::kind::lazy).submit({r}).wait();
}

void MKLDNNRequantizeForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  const RequantizeParam& param = nnvm::get<RequantizeParam>(attrs.parsed);
  float real_range;
  // model is calibrated
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    real_range =
          MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
    MKLDNNRequantizeForwardKer(attrs, ctx, inputs, req, outputs, real_range);
  // model is not calibrated
  } else {
    TShape src_shape, dst_shape;
    const size_t actual_float_size = sizeof(float);
    const size_t actual_quantized_size = sizeof(SrcDType);
    const size_t temp_reduce_size = ConfigReduce<cpu, SrcDType>(s,
                         inputs[0].shape_, TShape({1}), &src_shape, &dst_shape);
    Tensor<cpu, 1, char> temp_space =
      ctx.requested[0].get_space_typed<cpu, 1, char>(
      Shape1(2*actual_float_size+2*actual_quantized_size+temp_reduce_size), s);
    Tensor<cpu, 1, float> actual_min_float(
                 reinterpret_cast<float*>(temp_space.dptr_), Shape1(1), s);
    Tensor<cpu, 1, float> actual_max_float(
                 reinterpret_cast<float*>(temp_space.dptr_) + 1, Shape1(1), s);
    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob actual_min_quantized(reinterpret_cast<SrcDType*>(
                       temp_space.dptr_ + 8), Shape1(1), cpu::kDevMask, dev_id);
    TBlob actual_max_quantized(reinterpret_cast<SrcDType*>(
                   temp_space.dptr_ + 8) + 1, Shape1(1), cpu::kDevMask, dev_id);
    Tensor<cpu, 1, char> workspace(
            temp_space.dptr_+2*actual_float_size+2*actual_quantized_size,
            Shape1(temp_reduce_size), s);
    broadcast::Reduce<red::minimum, 2, SrcDType, mshadow::op::identity>(
        s, actual_min_quantized.reshape(dst_shape), kWriteTo,
        workspace, inputs[0].reshape(src_shape));
    Kernel<QuantizedToFloatStruct, cpu>::Launch(s, 1,
        actual_min_float.dptr_, actual_min_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());
    broadcast::Reduce<red::maximum, 2, SrcDType, mshadow::op::identity>(
        s, actual_max_quantized.reshape(dst_shape), kWriteTo,
        workspace, inputs[0].reshape(src_shape));
    Kernel<QuantizedToFloatStruct, cpu>::Launch(s, 1,
        actual_max_float.dptr_, actual_max_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());

    real_range = MaxAbs(*actual_min_float.dptr_, *actual_max_float.dptr_);
    MKLDNNRequantizeForwardKer(attrs, ctx, inputs, req, outputs, real_range);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_REQUANTIZE_INL_H_
