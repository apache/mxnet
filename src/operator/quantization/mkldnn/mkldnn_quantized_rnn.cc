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
 * Copyright (c) 2020 by Contributors
 * \file mkldnn_quantized_rnn.cc
 * \brief Common functions for quantized recurrent neural network
 * \author Zixuan Wei
*/

#if MXNET_USE_MKLDNN == 1

#include "../quantization_utils.h"
#include "./mkldnn_quantized_rnn-inl.h"

namespace mxnet {
namespace op {

std::vector<float> GetMKLDNNRnnWeightsQParams(const MKLDNNRnnFullParam& full_param,
                                              float* w_ptr) {
  const int nthreads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const RNNParam& default_param = full_param.default_param;
  const LayerParamVector& layer_params = full_param.layer_params;

  const MKLDNNRnnLayerParam& layer_param0 = layer_params.at(0);
  const size_t w_size0 = layer_param0.single_w_size;
  const size_t wx_size0 = 4 * layer_param0.state_size * layer_param0.input_size;
  const size_t wh_size0 = 4 * layer_param0.state_size * layer_param0.state_size;

  int directions = 1;
  float* wx = w_ptr;
  float* wh = wx + wx_size0;
  float* fake_wx = wx;
  float* fake_wh = wh;

  std::vector<float> wx_goi_max;
  std::vector<float> wh_goi_max;
  if (default_param.bidirectional) {
    directions = 2;
    wx_goi_max.resize(wx_size0);
    wh_goi_max.resize(wh_size0);
    fake_wx = wx_goi_max.data();
    fake_wh = wh_goi_max.data();
    #pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(wx_size0); ++i) {
      fake_wx[i] = MaxAbs(wx[i], wx[i + w_size0]);
    }
    #pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(wh_size0); ++i) {
      fake_wh[i] = MaxAbs(wh[i], wh[i + w_size0]);
    }
  }
  std::vector<float> w_max(4 * layer_param0.state_size, 0.0);
  const index_t input_size = layer_param0.input_size;          // input
  const index_t state_size = layer_param0.state_size;          // state
  const index_t gates_nblks = 4 * layer_param0.state_size;     // gates * state
  for (index_t go = 0; go < gates_nblks; ++go) {
    float tmp_max = w_max[go];
    for (index_t i = 0; i < input_size; ++i) {
      tmp_max = MaxAbs(fake_wx[go * input_size + i], tmp_max);
    }
    for (index_t i = 0; i < state_size; ++i) {
      tmp_max = MaxAbs(fake_wh[go * state_size + i], tmp_max);
    }
    w_max[go] = tmp_max;
  }
  wx += layer_param0.single_w_size * directions;
  wh += layer_param0.single_w_size * directions;

  std::vector<float> goi_max(wh_size0, 0.0);
  for (size_t lyr = 1; lyr < layer_params.size(); ++lyr) {
    const MKLDNNRnnLayerParam& layer_param = layer_params.at(lyr);
    const int weight_nblks = layer_param.num_layer * directions;
    for (int blk = 0; blk < weight_nblks; ++blk) {
      #pragma omp parallel for num_threads(nthreads)
      for (index_t i = 0; i < static_cast<index_t>(wh_size0); ++i) {
        goi_max[i] = MaxAbs(wx[i], wh[i]);
      }
      for (index_t go = 0; go < gates_nblks; ++go) {
        float tmp = w_max[go];
        //* NOTES: min/max reductions were supported since OpenMP 3.1, which was released in
        //  Jul 2011 (hence the version number).
        #if _OPENMP >= 201107
        #pragma omp parallel for reduction(max : tmp) num_threads(nthreads)
        #endif
        for (index_t i = 0; i < state_size; ++i) {
          tmp = Max(goi_max[go * state_size + i], tmp);
        }
        w_max[go] = tmp;
      }
    }
    wx += layer_param.single_w_size * directions;
    wh = wx + wh_size0;
  }
  #pragma omp parallel for num_threads(nthreads)
  for (index_t i = 0; i < static_cast<index_t>(w_max.size()); ++i) {
    w_max[i] = mshadow::red::limits::MaxValue<int8_t>() / w_max[i];
  }
  return w_max;
}

void MKLDNNQuantizedRnnOp::Forward(const OpContext &op_ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(op_ctx.requested[0]);
  const RNNParam &default_param = full_param_.default_param;
  const uint32_t num_base_inputs = GetRnnNumInputs(default_param);
  float data_scale = inputs[num_base_inputs + quantized_rnn::kDataScale].data().dptr<float>()[0];
  float data_shift = inputs[num_base_inputs + quantized_rnn::kDataShift].data().dptr<float>()[0];

  const NDArray &weights = inputs.at(rnn_enum::kParams);
  const size_t weights_size = weights.data().Size() -
      GetRnnBiasSize(default_param.num_layers, default_param.state_size,
      default_param.bidirectional + 1, default_param.mode);
  float *weights_ptr = weights.data().dptr<float>();
  float *bias_ptr = weights_ptr + weights_size;

  if (dmlc::GetEnv("MXNET_RNN_USE_WEIGHT_CACHE", 0) && !initialized_) {
    LOG(INFO) << "The current weight of RNN is assumed to be fixed and cached during "
        "the whole inference pipeline. Please set MXNET_RNN_USE_WEIGHT_CACHE=0, if "
        "the weight changed at runtime.";
  }
  const bool need_reset_weight = (!dmlc::GetEnv("MXNET_RNN_USE_WEIGHT_CACHE", 0) &&
      weights_ver_ != inputs[rnn_enum::kParams].version()) ? true : false;

  // Check if weights NDArray was changed. If so, reset initialized_
  if (!rnn_layers_.empty() &&
      ((cached_data_scale_ != data_scale || cached_data_shift_ != data_shift))) {
    initialized_ = false;
    weights_ver_ = inputs[rnn_enum::kParams].version();
    cached_data_scale_ = data_scale;
    cached_data_shift_ = data_shift;
  }

  if (!initialized_ || rnn_layers_.empty()) {
    weights_ver_ = inputs[rnn_enum::kParams].version();
    cached_data_scale_ = data_scale;
    cached_data_shift_ = data_shift;
    rnn_attr_->set_rnn_data_qparams(data_scale, data_shift);
    if (need_reset_weight || rnn_layers_.empty())
      rnn_attr_->set_rnn_weights_qparams(0 + (1 << 3) + (1 << 4),
          GetMKLDNNRnnWeightsQParams(full_param_, weights_ptr));
  }

  // Get data type
  const int data_dtype = outputs[rnn_enum::kOut].dtype();
  const int weights_dtype = inputs[rnn_enum::kParams].dtype();
  // Get temporary memory for output, state_out, statecell_out
  const int num_layers = default_param.num_layers;
  const int seq_length = default_param.seq_length_;
  const int batch_size = default_param.batch_size_;
  const int state_size = default_param.state_size;
  const int directions = default_param.bidirectional ? 2 : 1;
  mkldnn::memory::desc dst_desc({seq_length, batch_size, directions * state_size},
      get_mkldnn_type(data_dtype), mkldnn::memory::format_tag::tnc);
  mkldnn::memory::desc state_desc({num_layers, directions, batch_size, state_size},
      get_mkldnn_type(data_dtype), mkldnn::memory::format_tag::ldnc);
  auto out_mem = CreateMKLDNNMem(outputs[rnn_enum::kOut], dst_desc, req[rnn_enum::kOut]);
  mkldnn_output_t stateout_mem;
  mkldnn_output_t statecellout_mem;

  // Get input & output NDArray
  char *src = static_cast<char *>(inputs[rnn_enum::kData].data().dptr_);
  char *src_state = static_cast<char *>(inputs[rnn_enum::kState].data().dptr_);
  char *dst = static_cast<char *>(out_mem.second->get_data_handle());
  char *dst_state = nullptr;          // Output state
  char *src_state_cell = nullptr;     // Used in LSTM for cell state
  char *dst_state_cell = nullptr;     // Used in LSTM for cell state
  const size_t cell_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ *
      default_param.state_size * mshadow::mshadow_sizeof(data_dtype);

  const LayerParamVector& layer_params = full_param_.layer_params;
  for (size_t lyr = 0; lyr < layer_params.size(); ++lyr) {
    const MKLDNNRnnLayerParam& lyr_param = layer_params.at(lyr);
    const size_t single_w_size = lyr_param.single_w_size;
    const size_t native_single_b_size = lyr_param.native_single_b_size;
    const size_t directions = lyr_param.bidirectional ? 2 : 1;

    if (rnn_layers_.size() < layer_params.size()) {
      rnn_layers_.emplace_back(layer_params.at(lyr), false,
          inputs.at(quantized_rnn::kData), weights, rnn_attr_);
      rnn_layers_.back().SetWeightsMem(&(this->mgr_), weights_ptr, bias_ptr, false, weights_dtype);
    }
    MKLDNNRnnForward& rnn_layer = rnn_layers_.at(lyr);
    if (!initialized_ && rnn_layers_.size() == layer_params.size()) {
      rnn_layer.ResetFwd(inputs[rnn_enum::kData], inputs[rnn_enum::kParams], rnn_attr_);
    if (need_reset_weight) {
    }
      rnn_layer.SetWeightsMem(&(this->mgr_), weights_ptr, bias_ptr, false, weights_dtype);
    }
    weights_ptr += single_w_size * directions;
    bias_ptr += native_single_b_size * directions;

    if (default_param.state_outputs && req[rnn_enum::kStateOut] != kNullOp) {
      stateout_mem = CreateMKLDNNMem(
          outputs[rnn_enum::kStateOut], state_desc, req[rnn_enum::kStateOut]);
      dst_state = static_cast<char *>(stateout_mem.second->get_data_handle());
    }

    if (default_param.mode == rnn_enum::kLstm) {
      src_state_cell = static_cast<char *>(inputs[rnn_enum::kStateCell].data().dptr_);
      if (default_param.state_outputs && req[rnn_enum::kStateCellOut] != kNullOp) {
        statecellout_mem = CreateMKLDNNMem(
            outputs[rnn_enum::kStateCellOut], state_desc, req[rnn_enum::kStateCellOut]);
        dst_state_cell = static_cast<char *>(statecellout_mem.second->get_data_handle());
      }
    }
    src = lyr ? dst : src;
    rnn_layer.SetNewDataMem(src, src_state, src_state_cell,
        dst, dst_state, dst_state_cell, data_dtype);

    MKLDNNStream::Get()->RegisterPrimArgs(rnn_layer.GetFwd(), rnn_layer.GetArgsMap());

    if (lyr < default_param.num_layers - 1U) {
      src_state += cell_bytes;
      if (src_state_cell) src_state_cell += cell_bytes;
      if (dst_state) dst_state += cell_bytes;
      if (dst_state_cell) dst_state_cell += cell_bytes;
    }
  }
  initialized_ = true;
  CommitOutput(outputs[rnn_enum::kOut], out_mem);
  if (default_param.state_outputs) {
    CommitOutput(outputs[rnn_enum::kStateOut], stateout_mem);
    if (default_param.mode == rnn_enum::kLstm)
      CommitOutput(outputs[rnn_enum::kStateCellOut], statecellout_mem);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1