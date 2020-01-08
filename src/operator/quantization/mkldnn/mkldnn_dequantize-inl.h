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
 * \file mkldnn_dequantize-inl.h
 * \author Wenting Jiang, Xinyu Chen
 * \brief
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
#if MXNET_USE_MKLDNN == 1
#include <algorithm>
#include <string>
#include <vector>
#include "../../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {


class SgMKLDNNDequantizeOperator {
 public:
  explicit SgMKLDNNDequantizeOperator(const nnvm::NodeAttrs &attrs)
      : param_(nnvm::get<DequantizeParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req, const std::vector<NDArray> &outputs);

 private:
  bool initialized_{false};
  DequantizeParam param_;
  float cached_data_min_{0.f};
  float cached_data_max_{0.f};
  mkldnn::memory::desc o_desc_;
  mkldnn_args_map_t args_;
  std::shared_ptr<mkldnn::reorder> fwd_pd_;
};

void SgMKLDNNDequantizeOperator::Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &outputs) {
  NDArray in_buffer = inputs[0];
  if (inputs[0].IsView() && inputs[0].IsMKLDNNData()) in_buffer = inputs[0].Reorder2Default();
  auto i_mem = in_buffer.GetMKLDNNData();
  float data_min = *inputs[1].data().dptr<float>();
  float data_max = *inputs[2].data().dptr<float>();

  if (initialized_ && (cached_data_min_ != data_min || cached_data_max_ != data_max))
    initialized_ = false;

  if (!initialized_) {
    cached_data_min_ = data_min;
    cached_data_max_ = data_max;
    float real_range = MaxAbs(cached_data_min_, cached_data_max_);
    float quantized_range = 0.0;
    if (inputs[0].dtype() == mshadow::kUint8) {
      quantized_range = kUint8Range;
    } else if (inputs[0].dtype() == mshadow::kInt8) {
      quantized_range = kInt8Range;
      real_range = MaxAbs(*inputs[1].data().dptr<float>(), *inputs[2].data().dptr<float>());
    } else {
      LOG(FATAL) << "mkldnn dequantize op only supports int8 and uint8 as output type";
    }
    float scale = real_range / quantized_range;
    mkldnn::primitive_attr attr;
    const int mask = 0;
    std::vector<float> scales = {scale};
    attr.set_output_scales(mask, scales);
    mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
    auto i_desc = i_mem->get_desc();
    size_t i_ndim = in_buffer.shape().ndim();
    if (i_ndim == 4) {
      mkldnn::memory::format_tag o_fmt = mkldnn::memory::format_tag::nchw;
      mkldnn::memory::dims o_dims(i_desc.data.dims, i_desc.data.dims + i_desc.data.ndims);
      o_desc_ = mkldnn::memory::desc(o_dims, get_mkldnn_type<float>(), o_fmt);
    } else {
      o_desc_ = i_desc;
      o_desc_.data.data_type = get_mkldnn_type_t<float>();
    }
    auto reorder_pd =
        mkldnn::reorder::primitive_desc(cpu_engine, i_desc, cpu_engine, o_desc_, attr);
    fwd_pd_ = std::make_shared<mkldnn::reorder>(reorder_pd);
    initialized_ = true;
  }
  auto o_mem = CreateMKLDNNMem(outputs[0], o_desc_, req[0]);
  args_[MKLDNN_ARG_FROM] = *i_mem;
  args_[MKLDNN_ARG_TO] = *o_mem.second;
  MKLDNNStream::Get()->RegisterPrimArgs(*fwd_pd_, args_);
  CommitOutput(outputs[0], o_mem);
  MKLDNNStream::Get()->Submit();
}

static void SgMKLDNNDequantizeForward(const OpStatePtr &state_ptr, const OpContext &ctx,
                                      const std::vector<NDArray> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<NDArray> &outputs) {
  SgMKLDNNDequantizeOperator &op = state_ptr.get_state<SgMKLDNNDequantizeOperator>();
  op.Forward(ctx, inputs, req, outputs);
}



}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
