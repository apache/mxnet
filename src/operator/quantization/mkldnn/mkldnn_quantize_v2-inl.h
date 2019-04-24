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

class SgMKLDNNQuantizeOperator {
 public:
  explicit SgMKLDNNQuantizeOperator(const nnvm::NodeAttrs &attrs)
      : param_(nnvm::get<QuantizeV2Param>(attrs.parsed)) {}

  void Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req, const std::vector<NDArray> &outputs);

 private:
  bool initalized_{false};
  QuantizeV2Param param_;
  float cached_data_min_{0.f};
  float cached_data_max_{0.f};
  std::shared_ptr<mkldnn::memory> i_mem_;
  std::shared_ptr<mkldnn::memory> o_mem_;
  std::shared_ptr<mkldnn::reorder> fwd_pd_;
};

void SgMKLDNNQuantizeOperator::Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<NDArray> &outputs) {
  float quantized_range = 0.0;
  NDArray in_buffer = inputs[0];
  float data_min = mshadow::red::limits::MaxValue<float>();
  float data_max = mshadow::red::limits::MinValue<float>();

  // Pass through quantized data
  if (inputs[0].dtype() == mshadow::kUint8 || inputs[0].dtype() == mshadow::kInt8) {
    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      *outputs[1].data().dptr<float>() = param_.min_calib_range.value();
      *outputs[2].data().dptr<float>() = param_.max_calib_range.value();
    } else {
      if (inputs[0].dtype() == mshadow::kUint8) {
        *outputs[1].data().dptr<float>() = 0;
        *outputs[2].data().dptr<float>() = 255;
      } else {
        *outputs[1].data().dptr<float>() = -127;
        *outputs[2].data().dptr<float>() = 127;
      }
    }
    if (req[0] != kWriteInplace) {
      const_cast<NDArray &>(outputs[0]).CopyFrom(*inputs[0].GetMKLDNNData());
      MKLDNNStream::Get()->Submit();
    }
  } else {
    if (in_buffer.IsView() && in_buffer.IsMKLDNNData()) in_buffer = inputs[0].Reorder2Default();
    auto i_mem = in_buffer.GetMKLDNNData();

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      data_min = param_.min_calib_range.value();
      data_max = param_.max_calib_range.value();
    } else {
      // no calib info
      in_buffer = inputs[0].Reorder2Default();
      auto in_ptr = in_buffer.data().dptr<float>();
      auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
      std::vector<float> data_maxs(nthreads, data_max);
      std::vector<float> data_mins(nthreads, data_min);
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

      if (initalized_ && (cached_data_min_ != data_min || cached_data_max_ != data_max))
        initalized_ = false;
    }

    // Write output min/max
    auto out_type = GetQuantizeOutputType(param_);
    if (out_type == mshadow::kUint8) {
      quantized_range = kUint8Range;
      *outputs[1].data().dptr<float>() = data_min;
      *outputs[2].data().dptr<float>() = data_max;
    } else if (out_type == mshadow::kInt8) {
      float real_range = MaxAbs(data_min, data_max);
      quantized_range = kInt8Range;
      *outputs[1].data().dptr<float>() = -real_range;
      *outputs[2].data().dptr<float>() = real_range;
    } else {
      LOG(FATAL) << "mkldnn quantize op only supports int8 and uint8 as output type";
    }

    if (!initalized_) {
      cached_data_min_ = data_min;
      cached_data_max_ = data_max;
      float real_range = MaxAbs(data_min, data_max);
      float scale = quantized_range / real_range;
      primitive_attr attr;
      const int mask = 0;
      std::vector<float> scales = {scale};
      attr.set_output_scales(mask, scales);
      attr.set_int_output_round_mode(round_nearest);
      mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
      auto i_mpd = i_mem->get_primitive_desc();
      auto i_desc = i_mpd.desc();
      mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(i_desc.data.format);
      if (i_fmt == mkldnn::memory::format::nchw || i_fmt == mkldnn::memory::format::nChw8c ||
          i_fmt == mkldnn_nChw16c) {
        i_fmt = mkldnn::memory::format::nhwc;
      }
      size_t i_ndim = in_buffer.shape().ndim();
      mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
      for (size_t i = 0; i < i_ndim; i++) {
        i_dims[i] = static_cast<int>(in_buffer.shape()[i]);
      }
      auto o_desc = mkldnn::memory::desc(i_dims, get_mkldnn_type(out_type), i_fmt);
      auto o_mpd = memory::primitive_desc(o_desc, cpu_engine);
      auto reorder_pd = reorder::primitive_desc(i_mpd, o_mpd, attr);
      i_mem_ = std::make_shared<mkldnn::memory>(i_mpd, nullptr);
      o_mem_ = std::make_shared<mkldnn::memory>(o_mpd, nullptr);
      fwd_pd_ = std::make_shared<mkldnn::reorder>(reorder_pd, *i_mem_, *o_mem_);
      initalized_ = true;
    }
    auto o_mem = CreateMKLDNNMem(outputs[0], o_mem_->get_primitive_desc(), req[0]);
    i_mem_->set_data_handle(i_mem->get_data_handle());
    o_mem_->set_data_handle(o_mem.second->get_data_handle());
    MKLDNNStream::Get()->RegisterPrim(*fwd_pd_);
    CommitOutput(outputs[0], o_mem);
    MKLDNNStream::Get()->Submit();
  }
}

static void SgMKLDNNQuantizeForward(const OpStatePtr &state_ptr, const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  SgMKLDNNQuantizeOperator &op = state_ptr.get_state<SgMKLDNNQuantizeOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_V2_INL_H_
