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

static void MKLDNNQuantizedBatchNormForward(const nnvm::NodeAttrs& attrs, 
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &in_data,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &outputs) { 
  CHECK_EQ(in_data.size(), 7U);
  CHECK_EQ(outputs.size(), 3U);
  
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  const NDArray &data = in_data[quantized_batchnorm::kData];
  const size_t channelAxis = static_cast<size_t>(param.axis < 0
    ? static_cast<int>(data.shape().ndim()) + param.axis : param.axis);
  const int channel_count = data.shape()[channelAxis];
  const float min_data = in_data[quantized_batchnorm::kDataMin].data().dptr<float>()[0];
  const float max_data = in_data[quantized_batchnorm::kDataMax].data().dptr<float>()[0];
  const float max_abs_data = std::max(std::abs(min_data), std::abs(max_data));

  float *min_output_ptr = outputs[quantized_batchnorm::kOutMin].data().dptr<float>();
  float *max_output_ptr = outputs[quantized_batchnorm::kOutMax].data().dptr<float>();
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()){
    *max_output_ptr = std::max(std::abs(param.min_calib_range.value()), std::abs(param.max_calib_range.value()));
    *min_output_ptr = - *max_output_ptr;
  } else{
    LOG(FATAL) << "min_calib_range or max_calib_range is not available. Quantized BN currently don't support calib_mode=None";
  }

  unsigned flags = mkldnn::use_global_stats | mkldnn::use_scale_shift;
  auto &fwd = GetBNForward<float>(param, ctx, data, flags);
  const mkldnn::memory &weight_mem = fwd.GetWeight();
  CHECK_EQ(weight_mem.get_primitive_desc().get_size(), channel_count * sizeof(float) * 2);
  float* weight_buf = reinterpret_cast<float *>(weight_mem.get_data_handle()); 

  NDArray gamma = in_data[quantized_batchnorm::kGamma];
  NDArray beta = in_data[quantized_batchnorm::kBeta];
  float *gamma_ptr = gamma.data().dptr<float>();
  float *beta_ptr = beta.data().dptr<float>();

  NDArray moving_mean = in_data[quantized_batchnorm::kInMovingMean];
  NDArray moving_var = in_data[quantized_batchnorm::kInMovingVar];
  float *moving_mean_ptr = moving_mean.data().dptr<float>();
  float *moving_var_ptr = moving_var.data().dptr<float>();

  // rescale gamma and beta, to make mean=0 and var=1
  NDArray rescaled_mean = NDArray(moving_mean.storage_type(), moving_mean.shape(), moving_mean.ctx(), true, mshadow::kFloat32);
  NDArray rescaled_var = NDArray(moving_var.storage_type(), moving_var.shape(), moving_var.ctx(), true, mshadow::kFloat32);
  float *rescaled_mean_ptr = rescaled_mean.data().dptr<float>();
  float *rescaled_var_ptr = rescaled_var.data().dptr<float>();

  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int channel = 0; channel < channel_count; ++channel) {
    float invstd = 1.0 / std::sqrt(moving_var_ptr[channel] + param.eps);
    weight_buf[channel] = gamma_ptr[channel] * invstd * max_abs_data / (*max_output_ptr);
    weight_buf[channel_count + channel] = (beta_ptr[channel] - moving_mean_ptr[channel] * gamma_ptr[channel] * invstd)  * kInt8Range / (*max_output_ptr);
    rescaled_mean_ptr[channel] = 0.0f;
    rescaled_var_ptr[channel] = 1.0f;
  }
  
  const NDArray &out  = outputs[batchnorm::kOut];
  auto out_mem = const_cast<NDArray &>(out).CreateMKLDNNData(fwd.GetPd().dst_primitive_desc());
  fwd.SetDataHandle(data, rescaled_mean, rescaled_var, *out_mem);
                     
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
  MKLDNNStream::Get()->Submit();
}

inline static bool QuantizedBatchNormStorageType(const nnvm::NodeAttrs &attrs,
                                               const int dev_mask,
                                               DispatchMode *dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  bool dispatched = false;
  if (!dispatched) {
    dispatched = MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode,
                                   in_attrs, out_attrs);
  }
  if (!MKLDNNEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
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
