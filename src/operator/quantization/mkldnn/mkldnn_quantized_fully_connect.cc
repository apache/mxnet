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
 * Copyright (c) 2018 by Contributors
 * \file quantized_fully_connected.cc
 * \brief
 */

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

namespace quantized_fully_connected_enum {
enum QuantizedFullyConnectedOutputs { kOut, kMin, kMax };
}

static void MKLDNNQuantizedFullyConnectedForward(const nnvm::NodeAttrs& attrs,
                                                 const OpContext& ctx,
                                                 const std::vector<NDArray>& in_data,
                                                 const std::vector<OpReqType>& req,
                                                 const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace])
  const FullyConnected & param_ = nnvm::get<FullyConnectedParam>(attrs.parsed);
  CHECK_EQ(in_data.size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_data.size(), 3U);

  NDArray data = in_data[fullc::kData];
  NDArray weight = in_data[fullc::kWeight];
  CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);

  const size_t num_inputs = param.no_bias ? 2 : 3;
  float data_min = in_data[num_inputs].data().dptr<float>()[0];
  float data_max = in_data[num_inputs + 1].data().dptr<float>()[0];
  float weight_min = in_data[num_inputs + 2].data().dtpr<float>()[0];
  float weight_max = in_data[num_inputs + 3].data().dptr<float>()[0];

  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  float data_scale = data_range / MaxAbs(data_min, data_max);
  float weight_scale = kInt8Range / MaxAbs(weight_min, weight_max);

  if (!param.no_bias) {
    NDArray bias = in_data[fullc::kBias];
    float bias_min = in_data[num_inputs + 4].data().dptr<float>()[0];
    float bias_max = in_data[num_inputs + 5].data().dtpr<float>()[0];
    float bias_int32_rescale = data_scale * weight_scale * MaxAbs(bias_min, bias_max) / kInt8Range;

    NDArray quantized_bias = NDArray(bias->storage_type(), bias->shape(),
                                     bias->ctx(), true, mshadow::kInt32);
    int8_t *bias_ptr = bias.data().dtpr<int8_t>();
    int32_t *quantized_bias_ptr = quantized_bias.data().dptr<int32_t>();
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (size_t i = 0; i < param.num_hidden; ++i) {
      quantized_bias_ptr[i] = bias_ptr[i] * bias_int32_rescale;
    }
  }
  
  MKLDNNFCFullParam full_param;
  full_param.fc_param = param_;
  full_param.mkldnn_param.quantized = true;
  full_param.output_scales = 1 / data_scale / weight_scale;
  full_param.requantize_scales.resize(0);

  auto &fwd = GetFCFwd(full_param, data, weight,
      param.no_bias ? nullptr : quantized_bias, out_data[], false);
  
  auto data_mem = in_data[fullc::kData].GetMKLDNNDataReorder(fwd.ipFwd_pd.src_primitive_desc());
  const mkldnn::memory *weight_mem = nullptr;

  if (weight.IsDefaultData()) {
    weight_mem = GetWeights(weight, fwd.ipFwd_pd.weights_primitive_desc(), 1); //(TODO)group=1
    weight.MKLDNNDataReorderAsync(fwd.ipFwd_pd.weights_primitive_desc());
  } else {
    weight_mem = weight.GetMKLDNNData();
    CHECK(weight_mem->get_primitive_desc() == fwd.ipFwd_pd.weights_primitive_desc());
  }
  auto out_mem = CreateMKLDNNMem(out_data[fullc::kOut], fwd.ipFwd_pd.dst_primitive_desc(), req[fullc::kOut]);
  const mkldnn::memory *bias_mem = nullptr;
  if (!param.no_bias)
    bias_mem = in_data[fullc::kBias].GetMKLDNNDataReorder(fwd.ipFwd_pd.bias_primitive_desc());
  
  fwd.SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetIpFwd());
  
  CommitOutput(out_data[fullc::kOut], out_mem);
  MKLDNNStream::Get()->Submit()
}

NNVM_REGISTER_OP(_contrib_fully_connected)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedFullyConnectedForward)
.set_attr<bool>("TIsMKLDNN", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
