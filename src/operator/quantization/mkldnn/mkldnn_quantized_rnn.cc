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

/*!
 * \brief Quantization parameters of rnn weights' scales in an order of weights_qparams,
 *        weights_projection_qparams.
 */
typedef std::tuple<std::vector<float>, std::vector<float> > rnn_weights_qparams_t;

rnn_weights_qparams_t GetMKLDNNRnnWeightsQParams(
    const MKLDNNRnnFullParam& full_param, float* w_ptr) {
  const int nthreads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const RNNParam& default_param = full_param.default_param;
  const LayerParamVector& layer_params = full_param.layer_params;
  const bool use_proj = default_param.projection_size.has_value();
  const size_t iter_size = use_proj ?
      default_param.projection_size.value() : default_param.state_size;

  const MKLDNNRnnLayerParam& layer_param0 = layer_params.at(0);
  const size_t w_size0 = layer_param0.single_w_size;
  const size_t wx_size0 = 4 * layer_param0.state_size * layer_param0.input_size;
  const size_t wh_size0 = 4 * layer_param0.state_size * iter_size;
  const size_t wr_size = default_param.state_size * iter_size;

  int directions = 1;
  float* wx = w_ptr;
  float* wh = wx + wx_size0;
  float* wr = wh + wh_size0;
  float* fake_wx = wx;
  float* fake_wh = wh;
  float* fake_wr = wr;

  std::vector<float> wx_goi_max;
  std::vector<float> wh_goi_max;
  std::vector<float> wr_oi_max;
  if (default_param.bidirectional) {
    directions = 2;
    wx_goi_max.resize(wx_size0);
    wh_goi_max.resize(wh_size0);
    wr_oi_max.resize(wr_size);
    fake_wx = wx_goi_max.data();
    fake_wh = wh_goi_max.data();
    fake_wr = wr_oi_max.data();
    #pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(wx_size0); ++i) {
      fake_wx[i] = MaxAbs(wx[i], wx[i + w_size0]);
    }
    #pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(wh_size0); ++i) {
      fake_wh[i] = MaxAbs(wh[i], wh[i + w_size0]);
    }
    if (use_proj) {
      #pragma omp parallel for num_threads(nthreads)
      for (index_t i = 0; i < static_cast<index_t>(wr_size); ++i) {
        fake_wr[i] = MaxAbs(wr[i], wr[i + w_size0]);
      }
    }
  }
  std::vector<float> w_max(4 * layer_param0.state_size, 0.0);
  std::vector<float> proj_max(iter_size, 0.0);
  const index_t input_size = layer_param0.input_size;          // input
  const index_t state_size = layer_param0.state_size;          // state
  const index_t gates_nblks = 4 * layer_param0.state_size;     // gates * state
  for (index_t go = 0; go < gates_nblks; ++go) {
    float tmp_max = w_max[go];
    for (index_t i = 0; i < input_size; ++i) {
      tmp_max = MaxAbs(fake_wx[go * input_size + i], tmp_max);
    }
    for (index_t i = 0; i < static_cast<index_t>(iter_size); ++i) {
      tmp_max = MaxAbs(fake_wh[go * iter_size + i], tmp_max);
    }
    w_max[go] = tmp_max;
  }
  if (use_proj) {
    for (index_t i = 0; i < static_cast<index_t>(iter_size); ++i) {
      for (index_t s = 0; s < state_size; ++s) {
        proj_max[i] = MaxAbs(fake_wr[iter_size * state_size + s], proj_max[i]);
      }
    }
  }
  wx += layer_param0.single_w_size * directions;
  wh += layer_param0.single_w_size * directions;
  wr += layer_param0.single_w_size * directions;

  const size_t wx_size1 = 4 * default_param.state_size * default_param.state_size;
  const size_t wh_size1 = wh_size0;
  std::vector<float> go_max(gates_nblks, 0.0);
  for (size_t lyr = 1; lyr < layer_params.size(); ++lyr) {
    const MKLDNNRnnLayerParam& layer_param = layer_params.at(lyr);
    const int weight_nblks = layer_param.num_layer * directions;
    for (int blk = 0; blk < weight_nblks; ++blk) {
      for (index_t go = 0; go < gates_nblks; ++go) {
        float tmp = Abs(wx[0]);
        for (index_t i = 1; i < layer_param.input_size; ++i) {
          tmp = MaxAbs(wx[go * layer_param.input_size + i], tmp);
        }
        go_max[go] = Max(tmp, go_max[go]);
      }
      for (index_t go = 0; go < gates_nblks; ++go) {
        float tmp = Abs(wh[0]);
        for (index_t i = 1; i < static_cast<index_t>(iter_size); ++i) {
          tmp = MaxAbs(wh[go * iter_size + i], tmp);
        }
        go_max[go] = Max(tmp, go_max[go]);
      }
      #pragma omp parallel for num_threads(nthreads)
      for (index_t go = 0; go < gates_nblks; ++go) {
        w_max[go] = Max(go_max[go], w_max[go]);
      }
      if (use_proj) {
        for (index_t i = 0; i < static_cast<index_t>(iter_size); ++i) {
          for (index_t s = 0; s < state_size; ++s) {
            proj_max[i] = MaxAbs(fake_wr[iter_size * state_size + s], proj_max[i]);
          }
        }
      }
      wx += layer_param.single_w_size;
      wh = wx + wx_size1;
      wr = wh + wh_size1;
    }
  }
  #pragma omp parallel for num_threads(nthreads)
  for (index_t i = 0; i < static_cast<index_t>(w_max.size()); ++i) {
    w_max[i] = mshadow::red::limits::MaxValue<int8_t>() / w_max[i];
  }
  #pragma omp parallel for num_threads(nthreads)
  for (index_t i = 0; i < static_cast<index_t>(proj_max.size()); ++i) {
    proj_max[i] = mshadow::red::limits::MaxValue<int8_t>() / proj_max[i];
  }
  return std::make_tuple(w_max, proj_max);
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
    common::LogOnce("The current weight of RNN is assumed to be fixed and cached during "
        "the whole inference pipeline. Please set MXNET_RNN_USE_WEIGHT_CACHE=0, if "
        "the weight changed at runtime.");
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
    if (need_reset_weight || rnn_layers_.empty()) {
      rnn_weights_qparams_t weights_qparams =
          GetMKLDNNRnnWeightsQParams(full_param_, weights_ptr);
      rnn_attr_->set_rnn_weights_qparams(0 + (1 << 3) + (1 << 4), std::get<0>(weights_qparams));
      if (default_param.projection_size.has_value()) {
        rnn_attr_->set_rnn_weights_projection_qparams(0 + (1 << 3), std::get<1>(weights_qparams));
      }
    }
  }

  // Get data type
  const int data_dtype = outputs[rnn_enum::kOut].dtype();
  const int weights_dtype = inputs[rnn_enum::kParams].dtype();
  // Get temporary memory for output, state_out, statecell_out
  const int num_layers = default_param.num_layers;
  const int seq_length = default_param.seq_length_;
  const int batch_size = default_param.batch_size_;
  const int state_size = default_param.state_size;
  const int iter_size  = default_param.projection_size.has_value() ?
      default_param.projection_size.value() : state_size;
  const int directions = default_param.bidirectional ? 2 : 1;
  mkldnn::memory::desc dst_desc({seq_length, batch_size, directions * iter_size},
      get_mkldnn_type(data_dtype), mkldnn::memory::format_tag::tnc);
  mkldnn::memory::desc state_desc({num_layers, directions, batch_size, iter_size},
      get_mkldnn_type(data_dtype), mkldnn::memory::format_tag::ldnc);
  mkldnn::memory::desc cell_desc({num_layers, directions, batch_size, state_size},
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
  const size_t state_bytes = (default_param.bidirectional + 1) * batch_size *
      iter_size * mshadow::mshadow_sizeof(data_dtype);
  const size_t cell_bytes = (default_param.bidirectional + 1) * batch_size *
      state_size * mshadow::mshadow_sizeof(data_dtype);

  const LayerParamVector& layer_params = full_param_.layer_params;
  for (size_t lyr = 0; lyr < layer_params.size(); ++lyr) {
    const MKLDNNRnnLayerParam& lyr_param = layer_params.at(lyr);
    const size_t single_w_size = lyr_param.single_w_size;
    const size_t native_single_b_size = lyr_param.naive_single_b_size;
    const size_t directions = lyr_param.bidirectional ? 2 : 1;

    if (rnn_layers_.size() < layer_params.size()) {
      rnn_layers_.emplace_back(op_ctx.run_ctx.ctx, layer_params.at(lyr), false,
          inputs.at(quantized_rnn::kData), weights, rnn_attr_);
      rnn_layers_.back().SetWeightsMem(weights_ptr, bias_ptr, false, weights_dtype);
    }
    MKLDNNRnnForward& rnn_layer = rnn_layers_.at(lyr);
    if (!initialized_ && rnn_layers_.size() == layer_params.size()) {
      rnn_layer.ResetFwd(inputs[rnn_enum::kData], inputs[rnn_enum::kParams], rnn_attr_);
    }
    if (need_reset_weight) {
      rnn_layer.SetWeightsMem(weights_ptr, bias_ptr, false, weights_dtype);
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
            outputs[rnn_enum::kStateCellOut], cell_desc, req[rnn_enum::kStateCellOut]);
        dst_state_cell = static_cast<char *>(statecellout_mem.second->get_data_handle());
      }
    }
    src = lyr ? dst : src;
    rnn_layer.SetNewDataMem(src, src_state, src_state_cell,
        dst, dst_state, dst_state_cell, data_dtype);

    MKLDNNStream::Get()->RegisterPrimArgs(rnn_layer.GetFwd(), rnn_layer.GetArgsMap());

    if (lyr < default_param.num_layers - 1U) {
      src_state += state_bytes;
      if (dst_state) dst_state += state_bytes;
      if (src_state_cell) src_state_cell += cell_bytes;
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
