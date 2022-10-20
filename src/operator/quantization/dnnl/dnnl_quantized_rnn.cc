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
 * \file dnnl_quantized_rnn.cc
 * \brief Common functions for quantized recurrent neural network
 * \author Zixuan Wei
 */

#if MXNET_USE_ONEDNN == 1

#include "operator/quantization/quantization_utils.h"
#include "operator/quantization/dnnl/dnnl_quantized_rnn-inl.h"

namespace mxnet {
namespace op {

std::vector<float> GetDNNLRnnWeightsQParams(const DNNLRnnFullParam& full_param, float* w_ptr) {
  const int nthreads            = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const int num_gates           = 4;
  const RNNParam& default_param = full_param.default_param;
  const LayerParamVector& layer_params = full_param.layer_params;

  const DNNLRnnLayerParam& layer_param0 = layer_params.at(0);
  const size_t w_size0                  = layer_param0.single_w_size;
  const size_t wx_size0 = num_gates * layer_param0.state_size * layer_param0.input_size;
  const size_t wh_size0 = num_gates * layer_param0.state_size * layer_param0.state_size;

  int directions = 1;
  float* wx      = w_ptr;
  float* wh      = wx + wx_size0;
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
  std::vector<float> w_max(num_gates * layer_param0.state_size, 0.0);
  const index_t input_size  = layer_param0.input_size;              // input
  const index_t state_size  = layer_param0.state_size;              // state
  const index_t gates_nblks = num_gates * layer_param0.state_size;  // gates * state
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
    const DNNLRnnLayerParam& layer_param = layer_params.at(lyr);
    const int weight_nblks               = layer_param.num_layer * directions;
    for (int blk = 0; blk < weight_nblks; ++blk) {
#pragma omp parallel for num_threads(nthreads)
      for (index_t i = 0; i < static_cast<index_t>(wh_size0); ++i) {
        goi_max[i] = MaxAbs(wx[i], wh[i]);
      }
      for (index_t go = 0; go < gates_nblks; ++go) {
        float tmp = w_max[go];
// NOTES: min/max reductions were supported since OpenMP 3.1, which was
// released in Jul 2011 (hence the version number).
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

void DNNLQuantizedRnnOp::Init(const OpContext& op_ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  using format_tag = dnnl::memory::format_tag;

  // Get the bytes of a real type
  const Context& ctx            = op_ctx.run_ctx.ctx;
  const NDArray& weights        = inputs[rnn_enum::kParams];
  int dtype                     = weights.dtype();
  int weights_dtype             = weights.dtype();
  size_t dtype_bytes            = mshadow::mshadow_sizeof(dtype);
  const RNNParam& default_param = full_param_.default_param;
  const size_t weights_size =
      weights.data().Size() - GetRnnBiasSize(default_param.num_layers,
                                             default_param.state_size,
                                             default_param.bidirectional + 1,
                                             default_param.mode);
  char* weights_ptr = static_cast<char*>(weights.data().dptr_);
  char* bias_ptr    = weights_ptr + weights_size * dtype_bytes;

  // In the `autograd.record()` context, RNNOp is required to run into
  // `forward_training` mode.

  const size_t num_fusion = full_param_.layer_params.size();
  if (fwd_inf_vec_.size() < num_fusion) {
    size_t buffer_size = 0;  // Element number, instead of bytes, in the buffer
    for (auto& layer_param : full_param_.layer_params) {
      buffer_size += layer_param.workspace_size + layer_param.reserve_size;
    }
    buffer_size += outputs[rnn_enum::kOut].data().Size() * (num_fusion - 1);
    buffer_size += kDNNLAlign * num_fusion * 5;  // Add margin for alignment

    for (auto& layer_param : full_param_.layer_params) {
      fwd_inf_vec_.emplace_back(
          ctx, layer_param, false, inputs[rnn_enum::kData], inputs[rnn_enum::kParams], rnn_attr_);
      buffer_size += fwd_inf_vec_.back().GetSize();
    }
    mgr_.Init(buffer_size, ctx);
  }

  for (auto& fwd_layer : fwd_inf_vec_) {
    size_t single_w_bytes      = fwd_layer.GetParam().single_w_size * dtype_bytes;
    size_t single_b_bytes      = fwd_layer.GetParam().native_single_b_size * dtype_bytes;
    size_t directions          = fwd_layer.GetParam().bidirectional ? 2 : 1;
    size_t layer_weights_bytes = single_w_bytes * directions;
    size_t layer_bias_bytes    = single_b_bytes * directions;  // Native MXNet has double bias

    if (!fwd_layer.IsInitialized())
      fwd_layer.SetWeightsMem(weights_ptr, bias_ptr, false, weights_dtype);
    weights_ptr += layer_weights_bytes;
    bias_ptr += layer_bias_bytes;
  }

  CHECK_EQ(num_fusion, fwd_inf_vec_.size())
      << "Layer vector's size has a different value than the number of fusion.";
  if (dst_.size() < num_fusion - 1) {
    const int data_dtype = outputs[rnn_enum::kOut].dtype();
    // Here we need `fwd_inf_vec_.size() - 1` spaces for the intermediate
    // results of the multiple fused layers. And for the result of the last
    // fused layer, `outputs[rnn_enum::kOut]` could provide the space. Hence,
    // `forward_inf_vec_.back()` is excluded when allocates the spaces for
    // intermediate results.
    for (std::vector<DNNLRnnForward>::const_iterator fwd = fwd_inf_vec_.begin();
         fwd != fwd_inf_vec_.end() - 1;
         ++fwd)
      dst_.push_back(
          mgr_.Alloc({fwd->GetParam().dst_dims, get_dnnl_type(data_dtype), format_tag::tnc}));
  }

  initialized_ = true;
}

template <typename DNNLRnnX>
inline void RegisterDNNLRnn(DNNLRnnX const& rnn) {
  DNNLStream::Get()->RegisterPrimArgs(rnn.GetFwd(), rnn.GetArgsMap());
}

template <>
inline void RegisterDNNLRnn(DNNLRnnBackward const& rnn) {
  DNNLStream::Get()->RegisterPrimArgs(rnn.GetBwd(), rnn.GetArgsMap());
  rnn.SetNativeWeightsGrads();
}

void DNNLQuantizedRnnOp::Forward(const OpContext& op_ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(op_ctx.requested[0]);

  const RNNParam& default_param  = full_param_.default_param;
  const uint32_t num_base_inputs = GetRnnNumInputs(default_param);
  float data_scale = inputs[num_base_inputs + quantized_rnn::kDataScale].data().dptr<float>()[0];
  float data_shift = inputs[num_base_inputs + quantized_rnn::kDataShift].data().dptr<float>()[0];

  const bool need_reset_weight = (!dmlc::GetEnv("MXNET_RNN_USE_WEIGHT_CACHE", 0) &&
                                  weights_ver_ != inputs[rnn_enum::kParams].version()) ?
                                     true :
                                     false;
  const NDArray& weights = inputs.at(rnn_enum::kParams);
  float* weights_ptr     = weights.data().dptr<float>();
  if (!initialized_ || fwd_inf_vec_.empty()) {
    weights_ver_       = inputs[rnn_enum::kParams].version();
    cached_data_scale_ = data_scale;
    cached_data_shift_ = data_shift;
    rnn_attr_->set_rnn_data_qparams(data_scale, data_shift);
    if (need_reset_weight || fwd_inf_vec_.empty())
      rnn_attr_->set_rnn_weights_qparams(0 + (1 << 3) + (1 << 4),
                                         GetDNNLRnnWeightsQParams(full_param_, weights_ptr));
  }

  // Initialize weights version
  if (!initialized_ && weights_ver_ == 0) {
    weights_ver_       = inputs[rnn_enum::kParams].version();
    cached_data_scale_ = data_scale;
    cached_data_shift_ = data_shift;
  }

  if (!fwd_inf_vec_.empty() &&
      ((cached_data_scale_ != data_scale || cached_data_shift_ != data_shift))) {
    initialized_       = false;
    weights_ver_       = inputs[rnn_enum::kParams].version();
    cached_data_scale_ = data_scale;
    cached_data_shift_ = data_shift;
  }

  // Check if weights NDArray was changed. If so, reset initialized_
  if (fwd_inf_vec_.size() > 0 && weights_ver_ != inputs[rnn_enum::kParams].version()) {
    initialized_ = false;
    for (auto& fwd : fwd_inf_vec_)
      fwd.Reset();
    weights_ver_       = inputs[rnn_enum::kParams].version();
    cached_data_scale_ = data_scale;
    cached_data_shift_ = data_shift;
  }

  if (!initialized_ || fwd_inf_vec_.empty()) {
    Init(op_ctx, inputs, req, outputs);
  }

  // Get data type
  int data_dtype = outputs[rnn_enum::kOut].dtype();
  // Get temporary memory for output, state_out, statecell_out
  const int num_layers = default_param.num_layers;
  const int seq_length = default_param.seq_length_;
  const int batch_size = default_param.batch_size_;
  const int state_size = default_param.state_size;
  const int directions = default_param.bidirectional ? 2 : 1;
  dnnl::memory::desc dst_desc({seq_length, batch_size, directions * state_size},
                              get_dnnl_type(data_dtype),
                              dnnl::memory::format_tag::tnc);
  dnnl::memory::desc state_desc({num_layers, directions, batch_size, state_size},
                                get_dnnl_type(data_dtype),
                                dnnl::memory::format_tag::ldnc);
  auto out_mem = CreateDNNLMem(outputs[rnn_enum::kOut], dst_desc, req[rnn_enum::kOut]);
  dnnl_output_t stateout_mem;
  dnnl_output_t statecellout_mem;

  // Get input & output NDArray
  char* src               = static_cast<char*>(inputs[rnn_enum::kData].data().dptr_);
  char* src_state         = static_cast<char*>(inputs[rnn_enum::kState].data().dptr_);
  char* dst               = static_cast<char*>(out_mem.second->get_data_handle());
  char* dst_state         = nullptr;  // Output state
  char* src_state_cell    = nullptr;  // Used in LSTM for cell state
  char* dst_state_cell    = nullptr;  // Used in LSTM for cell state

  if (default_param.state_outputs && req[rnn_enum::kStateOut] != kNullOp) {
    stateout_mem =
        CreateDNNLMem(outputs[rnn_enum::kStateOut], state_desc, req[rnn_enum::kStateOut]);
    dst_state = static_cast<char*>(stateout_mem.second->get_data_handle());
  }

  if (default_param.mode == rnn_enum::kLstm) {
    src_state_cell = static_cast<char*>(inputs[rnn_enum::kStateCell].data().dptr_);
    if (default_param.state_outputs && req[rnn_enum::kStateCellOut] != kNullOp) {
      statecellout_mem =
          CreateDNNLMem(outputs[rnn_enum::kStateCellOut], state_desc, req[rnn_enum::kStateCellOut]);
      dst_state_cell = static_cast<char*>(statecellout_mem.second->get_data_handle());
    }
  }

  if (fwd_inf_vec_.size() == 1) {
    fwd_inf_vec_.front().SetNewDataMem(
        src, src_state, src_state_cell, dst, dst_state, dst_state_cell, data_dtype);
  } else {
    CHECK_EQ(fwd_inf_vec_.size(), dst_.size() + 1) << "Output memory error.";
    size_t cell_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ *
                        default_param.state_size * mshadow::mshadow_sizeof(data_dtype);

    // Set input data memory for the first layer. This stores intermediate
    // output results in this->xxx, used as the source input of the next layer.
    fwd_inf_vec_.front().SetNewDataMem(src,
                                       src_state,
                                       src_state_cell,
                                       this->dst_.front()->get_data_handle(),
                                       dst_state,
                                       dst_state_cell,
                                       data_dtype);
    // 1st_lyr -> dst_handle -> next_lyr -> dst_handle -> next_lyr -> ...
    for (size_t lyr = 1; lyr < fwd_inf_vec_.size() - 1; ++lyr) {
      src_state += cell_bytes;
      if (src_state_cell)
        src_state_cell += cell_bytes;
      if (dst_state)
        dst_state += cell_bytes;
      if (dst_state_cell)
        dst_state_cell += cell_bytes;
      fwd_inf_vec_.at(lyr).SetNewDataMem(this->dst_.at(lyr - 1)->get_data_handle(),
                                         src_state,
                                         src_state_cell,
                                         this->dst_.at(lyr)->get_data_handle(),
                                         dst_state,
                                         dst_state_cell,
                                         data_dtype);
    }
    // Set output data memory for the last layer.
    src_state += cell_bytes;
    if (src_state_cell)
      src_state_cell += cell_bytes;
    if (dst_state)
      dst_state += cell_bytes;
    if (dst_state_cell)
      dst_state_cell += cell_bytes;
    fwd_inf_vec_.back().SetNewDataMem(this->dst_.back()->get_data_handle(),
                                      src_state,
                                      src_state_cell,
                                      dst,
                                      dst_state,
                                      dst_state_cell,
                                      data_dtype);
  }

  for (auto& inf_lyr : fwd_inf_vec_)
    RegisterDNNLRnn(inf_lyr);

  CommitOutput(outputs[rnn_enum::kOut], out_mem);
  if (default_param.state_outputs) {
    CommitOutput(outputs[rnn_enum::kStateOut], stateout_mem);
    if (default_param.mode == rnn_enum::kLstm)
      CommitOutput(outputs[rnn_enum::kStateCellOut], statecellout_mem);
  }
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
