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
 * \file mkldnn_quantized_batch_norm.cc
 * \brief
 * \author Yixin Bao
*/

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_batch_norm-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

static void MKLDNNQuantizedBatchNormForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                            const std::vector<NDArray> &in_data,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<NDArray> &outputs) {
  CHECK_EQ(in_data.size(), 7U);
  CHECK_EQ(outputs.size(), 3U);

  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  const NDArray &data = in_data[quantized_batchnorm::kData];
  auto data_mem = data.GetMKLDNNData();

  // reorder if data type = uint8
  if (in_data[quantized_batchnorm::kData].dtype() == mshadow::kUint8) {
    auto u8_pd = data_mem->get_primitive_desc();
    auto u8_md = u8_pd.desc();
    mkldnn::memory::desc s8_md(
        mkldnn::memory::dims(u8_md.data.dims, u8_md.data.dims + u8_md.data.ndims),
        mkldnn::memory::data_type::s8, static_cast<mkldnn::memory::format>(u8_md.data.format));
    auto s8_pd = mkldnn::memory::primitive_desc(s8_md, CpuEngine::Get()->get_engine());
    auto data_reorder_mem = TmpMemMgr::Get()->Alloc(s8_pd);

    std::vector<float> reorder_scale;
    reorder_scale = {static_cast<float>(kInt8Range) / kUint8Range};
    primitive_attr reorder_attr;
    reorder_attr.set_int_output_round_mode(round_mode::round_nearest);
    reorder_attr.set_output_scales(0, reorder_scale);
    const auto reorder_pd = mkldnn::reorder::primitive_desc(u8_pd, s8_pd, reorder_attr);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *data_mem, *data_reorder_mem));
    data_mem = data_reorder_mem;
  }
  const size_t channelAxis = static_cast<size_t>(
      param.axis < 0 ? static_cast<int>(data.shape().ndim()) + param.axis : param.axis);
  const int channel_count = data.shape()[channelAxis];
  const float min_data = in_data[quantized_batchnorm::kDataMin].data().dptr<float>()[0];
  const float max_data = in_data[quantized_batchnorm::kDataMax].data().dptr<float>()[0];
  const float max_abs_data = std::max(std::abs(min_data), std::abs(max_data));

  float *min_output_ptr = outputs[quantized_batchnorm::kOutMin].data().dptr<float>();
  float *max_output_ptr = outputs[quantized_batchnorm::kOutMax].data().dptr<float>();
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    *max_output_ptr = param.max_calib_range.value();
    *min_output_ptr = param.min_calib_range.value();
  } else {
    LOG(FATAL) << "min_calib_range or max_calib_range is not available. Quantized BN currently "
                  "don't support calib_mode=None";
  }
  const float max_abs_output = std::max(std::abs(*min_output_ptr), std::abs(*max_output_ptr));

  unsigned flags = mkldnn::use_global_stats | mkldnn::use_scale_shift;
  auto &fwd = GetBNForward<float>(param, ctx, data_mem, flags);
  const mkldnn::memory &weight_mem = fwd.GetWeight();
  CHECK_EQ(weight_mem.get_primitive_desc().get_size(), channel_count * sizeof(float) * 2);
  float *weight_buf = reinterpret_cast<float *>(weight_mem.get_data_handle());

  float *gamma_ptr = in_data[quantized_batchnorm::kGamma].data().dptr<float>();
  float *beta_ptr = in_data[quantized_batchnorm::kBeta].data().dptr<float>();

  const NDArray &moving_mean = in_data[quantized_batchnorm::kInMovingMean];
  const NDArray &moving_var = in_data[quantized_batchnorm::kInMovingVar];
  float *moving_mean_ptr = moving_mean.data().dptr<float>();
  float *moving_var_ptr = moving_var.data().dptr<float>();

  // rescale gamma and beta, to make mean=0 and var=1
  auto rescaled_mean_mem =
      TmpMemMgr::Get()->Alloc(moving_mean.GetMKLDNNData()->get_primitive_desc());
  auto rescaled_var_mem = TmpMemMgr::Get()->Alloc(moving_var.GetMKLDNNData()->get_primitive_desc());
  float *rescaled_mean_ptr = reinterpret_cast<float *>(rescaled_mean_mem->get_data_handle());
  float *rescaled_var_ptr = reinterpret_cast<float *>(rescaled_var_mem->get_data_handle());

#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int channel = 0; channel < channel_count; ++channel) {
    float invstd = 1.0 / std::sqrt(moving_var_ptr[channel] + param.eps);
    weight_buf[channel] = gamma_ptr[channel] * invstd * max_abs_data / max_abs_output;
    weight_buf[channel_count + channel] =
        (beta_ptr[channel] - moving_mean_ptr[channel] * gamma_ptr[channel] * invstd) * kInt8Range /
        max_abs_output;
    rescaled_mean_ptr[channel] = 0.0f;
    rescaled_var_ptr[channel] = 1.0f;
  }

  auto out_mem = CreateMKLDNNMem(outputs[batchnorm::kOut],
      fwd.GetPd().dst_primitive_desc(), req[batchnorm::kOut], &data);
  fwd.SetDataHandle(data_mem, rescaled_mean_mem, rescaled_var_mem, out_mem.second);

  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
  MKLDNNStream::Get()->Submit();
}

inline static bool QuantizedBatchNormStorageType(const nnvm::NodeAttrs &attrs, const int dev_mask,
                                                 DispatchMode *dispatch_mode,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  bool dispatched = false;
  if (!dispatched) {
    dispatched = MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
  }
  return dispatched;
}

NNVM_REGISTER_OP(_contrib_quantized_batch_norm)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedBatchNormStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedBatchNormForward)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
