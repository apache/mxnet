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
  typedef int8_t  DstDType;
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
  float second_quantized_range = MinAbs(MaxValue<DstDType>(),
                                        MinValue<DstDType>());
  float second_scale = second_quantized_range / second_real_range;
  float scale = first_scale * second_scale;
  *outputs[1].data().dptr<float>() = -second_real_range;
  *outputs[2].data().dptr<float>() = second_real_range;
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
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_dim);
  for (size_t i = 0; i < i_dim; i++) {
    i_dims[i] = static_cast<int>(in_buffer.shape()[i]);
  }
  auto o_desc = mkldnn::memory::desc(i_dims,
                                    (mkldnn::memory::data_type)data_type_enum<DstDType>::type,
                                    i_fmt);
  auto o_mpd = memory::primitive_desc(o_desc, cpu_engine);
  auto reorder_pd  = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto o_mem = CreateMKLDNNMem(outputs[0], o_mpd, req[0]);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *i_mem, *o_mem.second));
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
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  const RequantizeParam& param = nnvm::get<RequantizeParam>(attrs.parsed);
  float real_range;
  // Model is calibrated
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    real_range =
          MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
    MKLDNNRequantizeForwardKer(attrs, ctx, inputs, req, outputs, real_range);
  // Model is not calibrated
  } else {
    TShape src_shape, dst_shape;
    const size_t actual_float_size = sizeof(float);
    const size_t actual_quantized_size = sizeof(SrcDType);
    const size_t temp_reduce_size = ConfigReduce<cpu, SrcDType>(s,
                         inputs[0].shape(), TShape({1}), &src_shape, &dst_shape);
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
        workspace, inputs[0].Reorder2Default().data().reshape(src_shape));
    Kernel<QuantizedToFloatStruct, cpu>::Launch(s, 1,
        actual_min_float.dptr_, actual_min_quantized.dptr<SrcDType>(),
        inputs[1].Reorder2Default().data().dptr<float>(),
        inputs[2].Reorder2Default().data().dptr<float>());
    broadcast::Reduce<red::maximum, 2, SrcDType, mshadow::op::identity>(
        s, actual_max_quantized.reshape(dst_shape), kWriteTo,
        workspace, inputs[0].Reorder2Default().data().reshape(src_shape));
    Kernel<QuantizedToFloatStruct, cpu>::Launch(s, 1,
        actual_max_float.dptr_, actual_max_quantized.dptr<SrcDType>(),
        inputs[1].Reorder2Default().data().dptr<float>(),
        inputs[2].Reorder2Default().data().dptr<float>());

    real_range = MaxAbs(*actual_min_float.dptr_, *actual_max_float.dptr_);
    MKLDNNRequantizeForwardKer(attrs, ctx, inputs, req, outputs, real_range);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_REQUANTIZE_INL_H_
