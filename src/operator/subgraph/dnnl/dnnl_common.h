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
 * \file dnnl_common.h
 * \brief Common header file for DNNL backend subgraph
 * \author Ciyong Chen
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_COMMON_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_COMMON_H_

#if MXNET_USE_ONEDNN == 1

#include <vector>

#include "operator/numpy/np_matrix_op-inl.h"

namespace mxnet {
namespace op {

template <typename DType>
static std::vector<float> GetWeightScales(const NDArray& weight,
                                          const NDArray* bias,
                                          const float data_scale,
                                          bool weight_channelwise_scale) {
  auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  std::vector<float> weight_scales;
  const DType* weight_ptr = weight.data().dptr<DType>();
  const DType* bias_ptr   = bias ? bias->data().dptr<DType>() : nullptr;
  const auto wshape       = weight.shape();
  size_t channel          = wshape[0];

  size_t offset = wshape.ProdShape(1, wshape.ndim());
  std::vector<DType> weight_c_min(channel, MaxValue<DType>());
  std::vector<DType> weight_c_max(channel, MinValue<DType>());
  for (int c = 0; c < static_cast<int>(channel); ++c) {
    const DType* p1 = weight_ptr + c * offset;
    for (size_t k = 0; k < offset; ++k) {
      if (weight_c_min[c] > p1[k])
        weight_c_min[c] = p1[k];
      if (weight_c_max[c] < p1[k])
        weight_c_max[c] = p1[k];
    }
  }

  if (weight_channelwise_scale) {
    weight_scales.resize(channel);
#pragma omp parallel for num_threads(nthreads)
    for (int c = 0; c < static_cast<int>(channel); ++c) {
      float scale = GetQuantizeScale(mshadow::kInt8, weight_c_min[c], weight_c_max[c]);
      if (bias_ptr && bias_ptr[c]) {
        // avoid overflow on bias
        // TODO(zhennan): dnnl has bug to handle INT_MAX in bias, so set the maximum value of bias
        // to INT_MAX / 2.
        float scale_max =
            static_cast<float>(bias_ptr[c] > 0 ? MaxValue<int32_t>() : MinValue<int32_t>()) / 2 /
            bias_ptr[c] / data_scale;
        scale = Min(scale, scale_max);
      }
      weight_scales[c] = scale;
    }
  } else {
    DType total_min = weight_c_min[0];
    DType total_max = weight_c_max[0];
    for (size_t c = 0; c < channel; ++c) {
      if (total_min > weight_c_min[c])
        total_min = weight_c_min[c];
      if (total_max < weight_c_max[c])
        total_max = weight_c_max[c];
    }
    weight_scales.resize(3);
    weight_scales[0] = GetQuantizeScale(mshadow::kInt8, total_min, total_max);
    weight_scales[1] = total_min;
    weight_scales[2] = total_max;
  }
  return weight_scales;
}

static inline void ConvertWeightBias2DNNL(NDArray* weight,
                                          NDArray* bias,
                                          bool has_bias,
                                          const dnnl::memory::desc& weight_md,
                                          const dnnl::memory::desc* bias_md,
                                          const int num_group,
                                          float data_scale,
                                          const std::vector<float>& weight_scales,
                                          const bool submit = true) {
  DNNLStream* stream             = DNNLStream::Get();
  const auto new_weight          = NDArray(&weight_md);
  const auto conv_weights_memory = new_weight.GetDNNLData();
  dnnl::primitive_attr weight_attr;
  if (weight_scales.size()) {
    const int weight_mask = (weight_scales.size()) == 1 ? 0 : 1;
    weight_attr.set_output_scales(weight_mask, weight_scales);
  }
  auto default_weights_memory = GetWeights(*weight, num_group);
  if (default_weights_memory == nullptr)
    default_weights_memory = weight->GetDNNLData();
  const auto weight_reorder_pd =
      dnnl::reorder::primitive_desc(*default_weights_memory, *conv_weights_memory, weight_attr);
  DNNLStream::Get()->RegisterPrimArgs(
      dnnl::reorder(weight_reorder_pd),
      {{DNNL_ARG_FROM, *default_weights_memory}, {DNNL_ARG_TO, *conv_weights_memory}});
  NDArray new_bias;
  if (has_bias && data_scale) {
    std::vector<float> bias_scales(weight_scales.size());
    for (size_t c = 0; c < weight_scales.size(); ++c) {
      bias_scales[c] = weight_scales[c] * data_scale;
    }
    new_bias                    = NDArray(bias_md);
    const auto conv_bias_memory = new_bias.GetDNNLData();
    const int bias_mask         = (bias_scales.size()) == 1 ? 0 : 1;
    dnnl::primitive_attr bias_attr;
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_weights_memory = bias->GetDNNLData();
    const auto bias_reorder_pd =
        dnnl::reorder::primitive_desc(*bias_weights_memory, *conv_bias_memory, bias_attr);
    DNNLStream::Get()->RegisterPrimArgs(
        dnnl::reorder(bias_reorder_pd),
        {{DNNL_ARG_FROM, *bias_weights_memory}, {DNNL_ARG_TO, *conv_bias_memory}});
  }
  if (submit)
    stream->Submit();
  *weight = new_weight;
  if (has_bias && data_scale)
    *bias = new_bias;
}

static inline bool CheckReshapeConditions(const nnvm::Node& node, const index_t out_index) {
  const index_t split_output_index = node.inputs[0].index;
  if (split_output_index != out_index)
    return false;

  const auto& reshape_param = nnvm::get<NumpyXReshapeParam>(node.attrs.parsed);
  const auto newshape       = reshape_param.newshape;

  if (newshape.ndim() != 4 || !(newshape[0] == newshape[1] && newshape[0] == -2))
    return false;

  return true;
}

static inline bool CheckSwapAxisConditions(const nnvm::Node& node) {
  auto params = node.attrs.dict;
  int dim1 = 0, dim2 = 0;
  if (params.count("dim1") && params.count("dim2")) {
    dim1 = std::stoi(params.at("dim1"));
    dim2 = std::stoi(params.at("dim2"));
  } else {
    return false;
  }

  return ((dim1 == 1 && dim2 == 2) || (dim1 == 2 && dim2 == 1));
}

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_COMMON_H_
