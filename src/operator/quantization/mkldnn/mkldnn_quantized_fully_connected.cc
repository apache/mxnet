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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_quantized_fully_connected.cc
 * \brief MKLDNN Quantized FullyConnected operator
 * \author Ciyong Chen
 */

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

void MKLDNNQuantizedFullyConnectedForward(const nnvm::NodeAttrs &attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &in_data,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  FullyConnectedParam param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const size_t num_inputs = param.no_bias ? 2 : 3;

  CHECK_EQ(in_data.size(), static_cast<size_t>(num_inputs * 3));
  CHECK_EQ(out_data.size(), 3U);

  NDArray data = in_data[fullc::kData];
  NDArray weight = in_data[fullc::kWeight];

  const float min_data =
    in_data[num_inputs + quantized_fullc::kDataMin].data().dptr<float>()[0];
  const float max_data =
    in_data[num_inputs + quantized_fullc::kDataMax].data().dptr<float>()[0];
  const float min_weight =
    in_data[num_inputs + quantized_fullc::kWeightMin].data().dptr<float>()[0];
  const float max_weight =
    in_data[num_inputs + quantized_fullc::kWeightMax].data().dptr<float>()[0];
  float *min_output_ptr = out_data[quantized_fullc::kOutMin].data().dptr<float>();
  float *max_output_ptr = out_data[quantized_fullc::kOutMax].data().dptr<float>();

  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  float data_scale = data_range / MaxAbs(min_data, max_data);
  float weight_scale = kInt8Range / MaxAbs(min_weight, max_weight);

  NDArray quantized_bias;
  if (!param.no_bias) {
    NDArray bias = in_data[fullc::kBias];
    float min_bias = in_data[num_inputs + quantized_fullc::kBiasMin].data().dptr<float>()[0];
    float max_bias = in_data[num_inputs + quantized_fullc::kBiasMax].data().dptr<float>()[0];
    float bias_int32_rescale = data_scale * weight_scale * MaxAbs(min_bias, max_bias) / kInt8Range;

    quantized_bias = NDArray(bias.storage_type(), bias.shape(),
                             bias.ctx(), true, mshadow::kInt32);
    int8_t *bias_ptr = bias.data().dptr<int8_t>();
    int32_t *quantized_bias_ptr = quantized_bias.data().dptr<int32_t>();
    size_t bias_size = bias.shape().Size();
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
      quantized_bias_ptr[i] = bias_ptr[i] * bias_int32_rescale;
    }
  }

  Stream<cpu> *s = ctx.get_stream<cpu>();
  if (data.dtype() == mshadow::kInt8) {
    mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
        s, 1, min_output_ptr, max_output_ptr, &min_data, &max_data, &min_weight, &max_weight);
  } else {
    mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
        s, 1, min_output_ptr, max_output_ptr, &min_data, &max_data, &min_weight, &max_weight);
  }

  bool is_train = false;
  mkldnn::memory::desc out_md = GetMemDesc(out_data[fullc::kOut]);
  MKLDNNFCFlattenData(param, out_data[fullc::kOut], &data, &out_md);
  auto &fwd = GetFCFwd(param, is_train, data, weight,
      param.no_bias ? nullptr : &quantized_bias, out_md);

  auto data_mem = in_data[fullc::kData].GetMKLDNNDataReorder(fwd.fwd_pd.src_desc());
  const mkldnn::memory *weight_mem = nullptr;

  if (weight.IsDefaultData()) {
    // We also need to modify the layout on the original weight array.
    // Don't switch below sequence because naive engine will executes
    // pushAsync synchronously.
    weight.MKLDNNDataReorderAsync(fwd.fwd_pd.weights_desc());
    weight_mem = GetWeights(weight, fwd.fwd_pd.weights_desc(), 1);
  } else {
    weight_mem = weight.GetMKLDNNData();
    CHECK(weight_mem->get_desc() == fwd.fwd_pd.weights_desc());
  }
  auto out_mem = CreateMKLDNNMem(out_data[fullc::kOut], fwd.fwd_pd.dst_desc(),
                                 req[fullc::kOut]);

  mkldnn_args_map_t args = {
      {MKLDNN_ARG_SRC, *data_mem},
      {MKLDNN_ARG_WEIGHTS, *weight_mem},
      {MKLDNN_ARG_DST, *out_mem.second},
  };

  const mkldnn::memory *bias_mem = nullptr;
  if (!param.no_bias) {
    bias_mem = quantized_bias.GetMKLDNNDataReorder(fwd.fwd_pd.bias_desc());
    args[MKLDNN_ARG_BIAS] = *bias_mem;
  }

  MKLDNNStream::Get()->RegisterPrimArgs(fwd.GetFwd(), args);
  CommitOutput(out_data[fullc::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
