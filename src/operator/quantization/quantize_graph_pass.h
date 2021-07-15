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
 *  Copyright (c) 2021 by Contributors
 * \file quantize_graph_pass.h
 * \brief
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_GRAPH_PASS_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_GRAPH_PASS_H_

#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "quantize_v2-inl.h"
#include "../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::ObjectPtr;
using nnvm::NodeEntry;
using nnvm::Graph;

inline ObjectPtr CreateNode(std::string op_name, std::string node_name) {
  ObjectPtr node = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed =
      nnvm::Symbol::CreateVariable(node->attrs.name).outputs[0].node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

template <bool require_bias>
static inline bool IsOneDNNFullyConnected(const ObjectPtr& n) {
#if MXNET_USE_MKLDNN == 1
  if (n->op() == Op::Get("_sg_mkldnn_fully_connected")) {
    auto const& param = nnvm::get<MKLDNNFCFullParam>(n->attrs.parsed);
    FCInputIndex idx(param);
    if (!(param.mkldnn_param.channel_wise_quantize.has_value() &&
          param.mkldnn_param.channel_wise_quantize.value())) {
      return !require_bias || (param.default_param.no_bias == false &&
                               n->inputs[idx.bias].node->is_variable());
    }
  }
#endif
  return false;
}

static inline bool IsQuantize(const ObjectPtr& n) {
  if (n->op() == Op::Get("_contrib_quantize_v2")) {
    auto const &param = nnvm::get<QuantizeV2Param>(n->attrs.parsed);
    if (param.min_calib_range.has_value() &&
        param.min_calib_range.value() < 0.0f) {
      return true;
    }
  }
  return false;
}

#if MXNET_USE_MKLDNN == 1
static NDArray* FindInArgByName(const Graph &g, const std::string& name) {
  const std::vector<std::string>& in_arg_names =
      g.GetAttr<std::vector<std::string>>("in_arg_names");
  size_t i = std::distance(in_arg_names.begin(),
                           std::find(in_arg_names.begin(), in_arg_names.end(), name));
  if (i == in_arg_names.size()) {
    LOG(FATAL) << name << " not found in in_arg_names";
  }
  return g.GetAttr<NDArray **>("in_args")[i];
}

// Rescales weights, min_weight and max_weight. Returns bias_int32_rescale.
static inline float RescaleWeights(const Graph &g, const ObjectPtr &fc, NDArray* weight_tensor) {
  aut& param = nnvm::get<MKLDNNFCFullParam>(fc->attrs.parsed);
  FCInputIndex idx(param);

  ObjectPtr &quantize = fc->inputs[idx.data].node;

  float* min_weight =
      FindInArgByName(g, fc->inputs[idx.weight_min].node->attrs.name)->data().dptr<float>();
  float* max_weight =
      FindInArgByName(g, fc->inputs[idx.weight_max].node->attrs.name)->data().dptr<float>();
  float min_bias =
      *FindInArgByName(g, fc->inputs[idx.bias_min].node->attrs.name)->data().dptr<float>();
  float max_bias =
      *FindInArgByName(g, fc->inputs[idx.bias_max].node->attrs.name)->data().dptr<float>();

  float min_data = param.mkldnn_param.min_calib_range.value();
  float max_data = param.mkldnn_param.max_calib_range.value();
  float data_scale_ = kUint8Range / (max_data - min_data);
  float weight_scale = GetQuantizeScale(mshadow::kInt8, *min_weight, *max_weight);
  float bias_scale = GetQuantizeScale(mshadow::kInt8, min_bias, max_bias);
  float bias_int32_rescale = data_scale_ * weight_scale / bias_scale;

  // // TODO(zhennan): mkldnn has bug to handle INT_MAX in bias, so set the
  // // maximum value of bias to INT_MAX / 2.
  float bias_max_rescale = mshadow::red::limits::MaxValue<int32_t>() / 2 /
                           MaxAbs(min_bias, max_bias) / bias_scale;
  if (bias_int32_rescale > bias_max_rescale) {
    LOG(INFO) << "RESCALING WEIGHTS in shifted quantization because bias scale "
                 "is too big in layer " << fc->attrs.name;
    // avoid overflow on bias
    bias_int32_rescale = bias_max_rescale;
    float weight_rescale =
        bias_int32_rescale * bias_scale / data_scale_ / weight_scale;

    size_t weight_size = weight_tensor->shape().Size();
    int8_t *weight_ptr = weight_tensor->data().dptr<int8_t>();
    for (int32_t i = 0; i < static_cast<int32_t>(weight_size); ++i) {
      weight_ptr[i] = std::round(weight_ptr[i] * weight_rescale);
    }
    *min_weight *= weight_rescale;
    *max_weight *= weight_rescale;
  }
  return bias_int32_rescale;
}

static inline void ShiftBias(int32_t* bias_ptr_int32, size_t bias_size,
                             NDArray* weight_tensor, int32_t shift_value) {
  CHECK_EQ(static_cast<size_t>(weight_tensor->shape()[0]), bias_size);
  int8_t* weight_ptr = weight_tensor->data().dptr<int8_t>();
  for (dim_t i = 0; i < weight_tensor->shape()[0]; ++i) {
    for (dim_t j = 0; j < weight_tensor->shape()[1]; j++) {
      bias_ptr_int32[i] -= shift_value * (*weight_ptr++);
    }
  }
}
#endif

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_GRAPH_PASS_H_
