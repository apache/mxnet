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

using nnvm::Symbol;
using nnvm::Node;
using nnvm::ObjectPtr;
using nnvm::NodeEntry;
using nnvm::Graph;

enum class Pattern {QuantizeFc, FcFc, None};

Pattern FindPattern(const ObjectPtr &node) {
  if (IsOneDNNFullyConnected<true>(node)) {
    if (IsQuantize(node->inputs[0].node)) {
      return Pattern::QuantizeFc;
    } else if (IsOneDNNFullyConnected<false>(node->inputs[0].node)) {
      return Pattern::FcFc;
    }
  }
  return Pattern::None;
}

void QuantizeFcShiftedQuantization(const ObjectPtr &node, Graph&& g,
                                   std::vector<NDArray *>* new_arg_vector,
                                   std::vector<std::string>* new_arg_names) {
  ObjectPtr &quantize = node->inputs[0].node;
  ObjectPtr& bias_node = node->inputs[2].node;
  std::string bias_name_old = bias_node->attrs.name;
  NDArray* bias_in_arg_ptr = FindInArgByName(g, bias_name_old);
  if (bias_in_arg_ptr->dtype() != mshadow::kInt8) return;
  std::string bias_name_s32 = bias_node->attrs.name + "_s32";
  bias_node = CreateNode("nullptr", bias_name_s32);
  new_arg_names->push_back(bias_name_s32);

  quantize->attrs.dict["shifted"] = "True";
  if (quantize->op()->attr_parser) quantize->op()->attr_parser(&(quantize->attrs));

  NDArray *weight_tensor = FindInArgByName(g, node->inputs[1].node->attrs.name);

  float bias_int32_rescale = RescaleWeights(g, node, weight_tensor);

  new_arg_vector->push_back(
      new NDArray(kDefaultStorage, bias_in_arg_ptr->shape(),
                  Context::CPU(), false, mshadow::kInt32));
  int32_t *bias_ptr_int32 = new_arg_vector->back()->data().dptr<int32_t>();
  size_t bias_size = bias_in_arg_ptr->shape().Size();
  int8_t *bias_ptr_old = bias_in_arg_ptr->data().dptr<int8_t>();

  for (size_t i = 0; i < bias_size; ++i) {
    bias_ptr_int32[i] = static_cast<int32_t>(
        std::round(bias_ptr_old[i] * bias_int32_rescale));
  }
  float min_data = std::stof(quantize->attrs.dict.at("min_calib_range"));
  float max_data = std::stof(quantize->attrs.dict.at("max_calib_range"));
  float data_scale = kUint8Range / (max_data - min_data);
  int32_t shift_value = static_cast<int32_t>(std::round(data_scale * -min_data));
  ShiftBias(bias_ptr_int32, bias_size, weight_tensor, shift_value);
  LOG(INFO) << "applied shifted quantization on QUANTIZE->FC";
}

void FcFcShiftedQuantization(const ObjectPtr &node, Graph&& g,
                             std::vector<NDArray *>* new_arg_vector,
                             std::vector<std::string>* new_arg_names) {
  ObjectPtr& first_fc = node->inputs[0].node;
  auto const& param =
      nnvm::get<MKLDNNFCFullParam>(first_fc->attrs.parsed);
  // TODO(sfraczek): remove !with_eltwise when onednn version upgrades
  if (!param.mkldnn_param.with_eltwise) {
    ObjectPtr& bias_node = node->inputs[2].node;
    std::string bias_name_old = bias_node->attrs.name;
    NDArray* bias_in_arg_ptr = FindInArgByName(g, bias_name_old);
    if (bias_in_arg_ptr->dtype() != mshadow::kInt8) return;
    std::string bias_name_s32 = bias_node->attrs.name + "_s32";
    bias_node = CreateNode("nullptr", bias_name_s32);
    new_arg_names->push_back(bias_name_s32);

    first_fc->attrs.dict["shifted_output"] = "True";
    if (first_fc->op()->attr_parser)
      first_fc->op()->attr_parser(&(first_fc->attrs));

    NDArray* weight_tensor =
        FindInArgByName(g, node->inputs[1].node->attrs.name);

    float bias_int32_rescale = RescaleWeights(g, node, weight_tensor);

    new_arg_vector->push_back(
        new NDArray(kDefaultStorage, bias_in_arg_ptr->shape(),
                    Context::CPU(), false, mshadow::kInt32));

    int32_t* bias_ptr_int32 =
        new_arg_vector->back()->data().dptr<int32_t>();
    size_t bias_size = bias_in_arg_ptr->shape().Size();
    int8_t* bias_ptr_old = bias_in_arg_ptr->data().dptr<int8_t>();

    for (size_t i = 0; i < bias_size; ++i) {
      bias_ptr_int32[i] = static_cast<int32_t>(
          std::round(bias_ptr_old[i] * bias_int32_rescale));
    }

    float min_data =
        std::stof(first_fc->attrs.dict.at("min_calib_range"));
    float max_data =
        std::stof(first_fc->attrs.dict.at("max_calib_range"));
    float data_scale = kUint8Range / (max_data - min_data);
    int32_t shift_value =
        static_cast<int32_t>(std::round(data_scale * -min_data));
    ShiftBias(bias_ptr_int32, bias_size, weight_tensor, shift_value);
    LOG(INFO) << "applied shifted quantization on FC->FC";
  }
}

Graph OneDNNShiftedQuantization(Graph&& g) {
  bool disable_shifted_quant =
      dmlc::GetEnv("MXNET_DISABLE_SHIFTED_QUANTIZATION_OPTIMIZATIONS", true);
  bool quantize_fc = !dmlc::GetEnv("MXNET_DISABLE_SHIFTED_QUANTIZE_FC_OPTIMIZATION", false);
  bool fc_fc = !dmlc::GetEnv("MXNET_DISABLE_SHIFTED_FC_FC_OPTIMIZATION", false);
  LOG(INFO) << "Running OneDNN shifted quantization: " << !disable_shifted_quant;
  // No change to aux params
  g.attrs["new_aux_names"] = std::make_shared<nnvm::any>(std::vector<std::string>());
  g.attrs["new_aux"] = std::make_shared<nnvm::any>(std::vector<NDArray *>());

  // New args to replace the old
  std::vector<std::string> new_arg_names;
  std::vector<NDArray *> new_arg_vector;

  if (!disable_shifted_quant) {
    DFSVisit(g.outputs, [&](const ObjectPtr &node) {
      Pattern p = FindPattern(node);
      switch (p) {
        case Pattern::QuantizeFc:
          if (quantize_fc) {
            QuantizeFcShiftedQuantization(node, std::forward<Graph>(g),
                                          &new_arg_vector, &new_arg_names);
          }
          break;
        case Pattern::FcFc:
          if (fc_fc) {
            FcFcShiftedQuantization(node, std::forward<Graph>(g),
                                    &new_arg_vector, &new_arg_names);
          }
          break;
      }
    });
  }
  g.attrs["new_arg_names"] = std::make_shared<nnvm::any>(new_arg_names);
  g.attrs["new_args"] = std::make_shared<nnvm::any>(new_arg_vector);
  return g;
}

NNVM_REGISTER_PASS(OneDNNShiftedQuantization)
.describe("Enables shifted quantization.")
.set_body(OneDNNShiftedQuantization)
.set_change_graph(true);

}  // namespace asym_quant
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
