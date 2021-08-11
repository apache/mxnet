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
 * \file asymmetric_quantize_graph_pass.cc
 * \brief
 */
#if MXNET_USE_MKLDNN == 1
#include "quantize_graph_pass.h"

namespace mxnet {
namespace op {
namespace asym_quant {

using nnvm::Graph;
using nnvm::ObjectPtr;

template <bool require_bias>
static bool IsMKLDNNFullyConnected(const ObjectPtr& n) {
  if (n->op() == Op::Get("_sg_mkldnn_fully_connected")) {
    auto const& param = nnvm::get<MKLDNNFCFullParam>(n->attrs.parsed);
    FCInputIndex idx(param);
    if (!(param.mkldnn_param.channel_wise_quantize.has_value() &&
          param.mkldnn_param.channel_wise_quantize.value())) {
      return !require_bias ||
             (param.default_param.no_bias == false && n->inputs[idx.bias].node->is_variable());
    }
  }
  return false;
}

static bool IsQuantize(const ObjectPtr& n) {
  if (n->op() == Op::Get("_contrib_quantize_v2")) {
    auto const& param = nnvm::get<QuantizeV2Param>(n->attrs.parsed);
    if (param.min_calib_range.has_value() && param.min_calib_range.value() < 0.0f) {
      return true;
    }
  }
  return false;
}

static NDArray* FindInArgByName(const Graph& g, const std::string& name) {
  const std::vector<std::string>& in_arg_names =
      g.GetAttr<std::vector<std::string>>("in_arg_names");
  size_t i = std::distance(in_arg_names.begin(),
                           std::find(in_arg_names.begin(), in_arg_names.end(), name));
  if (i == in_arg_names.size()) {
    LOG(FATAL) << name << " not found in in_arg_names";
  }
  return g.GetAttr<NDArray**>("in_args")[i];
}

// Rescales weights, min_weight and max_weight. Returns bias_int32_rescale.
static float RescaleWeights(const Graph& g,
                            const ObjectPtr& fc,
                            NDArray* weight_tensor,
                            float min_data,
                            float max_data,
                            FCInputIndex idx) {
  auto fc_input_node_name = [&fc](int input) { return fc->inputs[input].node->attrs.name; };

  float* min_weight = FindInArgByName(g, fc_input_node_name(idx.weight_min))->data().dptr<float>();
  float* max_weight = FindInArgByName(g, fc_input_node_name(idx.weight_max))->data().dptr<float>();
  float min_bias    = *FindInArgByName(g, fc_input_node_name(idx.bias_min))->data().dptr<float>();
  float max_bias    = *FindInArgByName(g, fc_input_node_name(idx.bias_max))->data().dptr<float>();

  float data_scale_        = kUint8Range / (max_data - min_data);
  float weight_scale       = GetQuantizeScale(mshadow::kInt8, *min_weight, *max_weight);
  float bias_scale         = GetQuantizeScale(mshadow::kInt8, min_bias, max_bias);
  float bias_int32_rescale = data_scale_ * weight_scale / bias_scale;

  // // TODO(zhennan): mkldnn has bug to handle INT_MAX in bias, so set the
  // // maximum value of bias to INT_MAX / 2.
  float bias_max_rescale =
      mshadow::red::limits::MaxValue<int32_t>() / 2 / MaxAbs(min_bias, max_bias) / bias_scale;
  if (bias_int32_rescale > bias_max_rescale) {
    LOG(INFO) << "RESCALING WEIGHTS in shifted quantization because bias scale "
                 "is too big in layer "
              << fc->attrs.name;
    // avoid overflow on bias
    bias_int32_rescale   = bias_max_rescale;
    float weight_rescale = bias_int32_rescale * bias_scale / data_scale_ / weight_scale;

    size_t weight_size = weight_tensor->shape().Size();
    int8_t* weight_ptr = weight_tensor->data().dptr<int8_t>();
    for (int32_t i = 0; i < static_cast<int32_t>(weight_size); ++i) {
      weight_ptr[i] = std::round(weight_ptr[i] * weight_rescale);
    }
    *min_weight *= weight_rescale;
    *max_weight *= weight_rescale;
  }
  return bias_int32_rescale;
}

static void ShiftBias(int32_t* bias_ptr_int32,
                      size_t bias_size,
                      NDArray* weight_tensor,
                      int32_t shift_value) {
  CHECK_EQ(static_cast<size_t>(weight_tensor->shape()[0]), bias_size);
  int8_t* weight_ptr = weight_tensor->data().dptr<int8_t>();
  for (dim_t i = 0; i < weight_tensor->shape()[0]; ++i) {
    for (dim_t j = 0; j < weight_tensor->shape()[1]; j++) {
      bias_ptr_int32[i] -= shift_value * (*weight_ptr++);
    }
  }
}

enum class Pattern { QuantizeFc, FcFc, None };

static Pattern FindPattern(const ObjectPtr& node) {
  if (IsMKLDNNFullyConnected<true>(node)) {
    if (IsQuantize(node->inputs[0].node)) {
      return Pattern::QuantizeFc;
    } else if (IsMKLDNNFullyConnected<false>(node->inputs[0].node)) {
      return Pattern::FcFc;
    }
  }
  return Pattern::None;
}

static void FCShiftedQuantization(const ObjectPtr& node,
                                  const Graph& g,
                                  std::vector<NDArray*>* new_arg_vector,
                                  std::vector<std::string>* new_arg_names,
                                  const char* attr_name) {
  FCInputIndex idx(nnvm::get<MKLDNNFCFullParam>(node->attrs.parsed));

  ObjectPtr& bias_node      = node->inputs[idx.bias].node;
  std::string bias_name_old = bias_node->attrs.name;
  NDArray* bias_in_arg_ptr  = FindInArgByName(g, bias_name_old);
  if (bias_in_arg_ptr->dtype() != mshadow::kInt8)
    return;
  std::string bias_name_s32 = bias_node->attrs.name + "_s32";
  bias_node                 = CreateNode("nullptr", bias_name_s32);
  new_arg_names->push_back(bias_name_s32);

  ObjectPtr& input_node             = node->inputs[idx.data].node;
  input_node->attrs.dict[attr_name] = "True";
  if (input_node->op()->attr_parser)
    input_node->op()->attr_parser(&(input_node->attrs));

  float min_data           = std::stof(input_node->attrs.dict.at("min_calib_range"));
  float max_data           = std::stof(input_node->attrs.dict.at("max_calib_range"));
  NDArray* weight_tensor   = FindInArgByName(g, node->inputs[1].node->attrs.name);
  float bias_int32_rescale = RescaleWeights(g, node, weight_tensor, min_data, max_data, idx);

  new_arg_vector->push_back(new NDArray(
      kDefaultStorage, bias_in_arg_ptr->shape(), Context::CPU(), false, mshadow::kInt32));

  int32_t* bias_ptr_int32 = new_arg_vector->back()->data().dptr<int32_t>();
  size_t bias_size        = bias_in_arg_ptr->shape().Size();
  int8_t* bias_ptr_old    = bias_in_arg_ptr->data().dptr<int8_t>();

  for (size_t i = 0; i < bias_size; ++i) {
    bias_ptr_int32[i] = static_cast<int32_t>(std::round(bias_ptr_old[i] * bias_int32_rescale));
  }

  float data_scale    = kUint8Range / (max_data - min_data);
  int32_t shift_value = static_cast<int32_t>(std::round(data_scale * -min_data));
  ShiftBias(bias_ptr_int32, bias_size, weight_tensor, shift_value);
}

static Graph MKLDNNShiftedQuantization(Graph&& g) {
  bool disable_shifted_quant =
      dmlc::GetEnv("MXNET_DISABLE_SHIFTED_QUANTIZATION_OPTIMIZATIONS", true);
  bool quantize_fc = !dmlc::GetEnv("MXNET_DISABLE_SHIFTED_QUANTIZE_FC_OPTIMIZATION", false);
  bool fc_fc       = !dmlc::GetEnv("MXNET_DISABLE_SHIFTED_FC_FC_OPTIMIZATION", false);
  if (!disable_shifted_quant) {
    LOG(INFO) << "Running MKLDNN shifted quantization";
  }
  // No change to aux params
  g.attrs["new_aux_names"] = std::make_shared<nnvm::any>(std::vector<std::string>());
  g.attrs["new_aux"]       = std::make_shared<nnvm::any>(std::vector<NDArray*>());

  // New args to replace the old
  std::vector<std::string> new_arg_names;
  std::vector<NDArray*> new_arg_vector;

  if (!disable_shifted_quant) {
    unsigned quantize_fc_counter = 0;
    unsigned fc_fc_counter       = 0;
    DFSVisit(g.outputs, [&](const ObjectPtr& node) {
      Pattern p = FindPattern(node);
      switch (p) {
        case Pattern::QuantizeFc:
          if (quantize_fc) {
            FCShiftedQuantization(node, g, &new_arg_vector, &new_arg_names, "shifted");
            ++quantize_fc_counter;
          }
          break;
        case Pattern::FcFc:
          if (fc_fc) {
            FCShiftedQuantization(node, g, &new_arg_vector, &new_arg_names, "shifted_output");
            ++fc_fc_counter;
          }
          break;
        default:
          break;
      }
    });
    if (quantize_fc_counter > 0) {
      LOG(INFO) << "applied shifted quantization on QUANTIZE->FC " << quantize_fc_counter
                << " times";
    }
    if (fc_fc_counter > 0) {
      LOG(INFO) << "applied shifted quantization on FC->FC " << fc_fc_counter << " times";
    }
  }
  g.attrs["new_arg_names"] = std::make_shared<nnvm::any>(new_arg_names);
  g.attrs["new_args"]      = std::make_shared<nnvm::any>(new_arg_vector);
  return g;
}

NNVM_REGISTER_PASS(MKLDNNShiftedQuantization)
    .describe("Enables shifted quantization.")
    .set_body(MKLDNNShiftedQuantization)
    .set_change_graph(true);

}  // namespace asym_quant
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
