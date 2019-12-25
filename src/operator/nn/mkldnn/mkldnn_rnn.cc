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
 * \file mkldnn_rnn.cc
 * \brief Common functions used by MKLDNN RNN operator
 * \author Zixuan Wei
*/

#if MXNET_USE_MKLDNN == 1

#include <numeric>
#include "./mkldnn_rnn-inl.h"

namespace mxnet {
namespace op {

inline int GetRnnGatesNum(int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      return 4;
    case rnn_enum::kGru:
      return 3;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      return 1;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      return -1;
  }
}

void MKLDNNRnnLayerParam::SetDims() {
  const int ngates = GetRnnGatesNum(mode);
  //* NOTES: LBR-GRU's new gate formula needs two bias. So it has one more bias with LBR-GRU
  const int nbias = mode == rnn_enum::kGru ? (ngates + 1) : ngates;
  const int num_direction = bidirectional ? 2 : 1;

  src_dims.assign({seq_len, batch_size, input_size});
  weight_layer_dims.assign({num_layer, num_direction, input_size, ngates, state_size});
  weight_iter_dims.assign({num_layer, num_direction, state_size, ngates, state_size});
  bias_dims.assign({num_layer, num_direction, nbias, state_size});
  dst_dims.assign({seq_len, batch_size, state_size * num_direction});
  state_dims.assign({num_layer, num_direction, batch_size, state_size});

  // unidirectional size of a single cell
  single_w_size = (input_size + state_size) * ngates * state_size;
  single_b_size = nbias * state_size;
  naive_single_b_size = ngates * state_size * 2;  // naive RNN variants have double bias
  single_state_size = batch_size * state_size;

  // Get workspace size for cached weights memory
  // multiplication of tensor dimensions
  static auto tz_volume = [](const memory::dims& tz_dims) {
    return std::accumulate(tz_dims.begin(), tz_dims.end(), static_cast<memory::dim>(1),
        std::multiplies<memory::dim>());
  };

  workspace_size = tz_volume(weight_layer_dims) + tz_volume(weight_iter_dims) +
      tz_volume(bias_dims);
  reserve_size = 0;
}

MKLDNNRnnFullParam MKLDNNRnnFullParamParser(const RNNParam& rnn_param, const int seq_len,
                                            const int batch_size, const int input_size) {
  MKLDNNRnnFullParam full_param;
  full_param.default_param = rnn_param;
  size_t state_size = rnn_param.state_size;
  LayerParamVector &layer_params = full_param.layer_params;

  full_param.default_param.seq_length_ = seq_len;
  full_param.default_param.batch_size_ = batch_size;
  full_param.default_param.input_size_ = input_size;
  // Set basic size by constructing MKLDNNRnnLayerParam instance(s)
  if (rnn_param.bidirectional) {  // unfused bidirectional multi-layer RNN
    layer_params.emplace_back(1, batch_size, seq_len, input_size, state_size, rnn_param.mode);
    for (size_t layer = 1; layer < rnn_param.num_layers; ++layer) {
      layer_params.emplace_back(1, batch_size, seq_len, state_size * 2, state_size,
          rnn_param.mode);
    }
  } else if (input_size == static_cast<int>(state_size)) {  // fused multi-layer RNN
    layer_params.emplace_back(rnn_param.num_layers, batch_size, seq_len, input_size,
        state_size, rnn_param.mode, false);
  } else {  // unfused 1st layer, plus fused 2-end layers
    layer_params.emplace_back(1, batch_size, seq_len, input_size, state_size, rnn_param.mode,
        false);
    if (rnn_param.num_layers > 1)
      layer_params.emplace_back(rnn_param.num_layers - 1, batch_size, seq_len, state_size,
          state_size, rnn_param.mode, false);
  }

  // Set dims, workspace size, and state_outputs flag
  for (auto& layer_param : layer_params) {
    layer_param.SetDims();
    layer_param.state_outputs = rnn_param.state_outputs;
  }
  return full_param;
}

void MKLDNNRnnMemMgr::Init(dim_t size, const Context& ctx, int dtype) {
  workspace_ = NDArray(TShape({size}), ctx, false, dtype);
  curr_mem = static_cast<char *>(workspace_.data().dptr_);
  mem_size = size * mshadow::mshadow_sizeof(dtype);
  curr_size = size * mshadow::mshadow_sizeof(dtype);
}

mkldnn::memory *MKLDNNRnnMemMgr::Alloc(const mkldnn::memory::desc &md) {
  if (curr_mem == nullptr) {
    curr_mem = static_cast<char *>(workspace_.data().dptr_);
  }

  mkldnn_mem_ptr ret(new mkldnn::memory());
  size_t addr = reinterpret_cast<size_t>(curr_mem);
  size_t last_chunk = addr % alignment;
  size_t padding = alignment - last_chunk;
  addr += padding;
  CHECK_EQ(addr % alignment, 0);

  curr_size -= (md.get_size() + padding);
  if (curr_size < 0) {
    ret.reset(new mkldnn::memory(md, cpu_engine));
  } else {
    curr_mem += (md.get_size() + padding);
    ret.reset(new mkldnn::memory(md, cpu_engine, reinterpret_cast<void *>(addr)));
  }
  RegisterMem(ret);
  return ret.get();
}

RnnPrimitive GetRnnFwdPrim(
    const MKLDNNRnnLayerParam &layer_param, const bool is_train,
    const NDArray &data, const NDArray &params) {
  using namespace mkldnn;
  using tag = mkldnn::memory::format_tag;
  const int mode = layer_param.mode;
  memory::data_type data_type = get_mkldnn_type(data.dtype());
  memory::data_type weight_type = get_mkldnn_type(params.dtype());
  const prop_kind prop = is_train ? prop_kind::forward_training : prop_kind::forward_inference;
  const rnn_direction mkldnn_rnn_direction = layer_param.bidirectional ?
      rnn_direction::bidirectional_concat : rnn_direction::unidirectional;

  auto src_layer_desc    = memory::desc(layer_param.src_dims, data_type, tag::tnc);
  auto weight_layer_desc = memory::desc(layer_param.weight_layer_dims, weight_type, tag::any);
  auto weight_iter_desc  = memory::desc(layer_param.weight_iter_dims, weight_type, tag::any);
  auto bias_desc         = memory::desc(layer_param.bias_dims, data_type, tag::ldgo);
  auto dst_layer_desc    = memory::desc(layer_param.dst_dims, data_type, tag::tnc);
  auto src_state_desc    = memory::desc(layer_param.state_dims, data_type, tag::ldnc);
  auto dst_state_desc = layer_param.state_outputs ? memory::desc(
      layer_param.state_dims, data_type, tag::ldnc) : memory::desc();

  auto fwd = RnnPrimitive();
  switch (mode) {
    case rnn_enum::kLstm:
      fwd = RnnPrimitive::Create<lstm_forward>(prop, mkldnn_rnn_direction,
          src_layer_desc, src_state_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc,
          dst_state_desc);
      break;
    case rnn_enum::kGru:
      fwd = RnnPrimitive::Create<lbr_gru_forward>(prop, mkldnn_rnn_direction,
          src_layer_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc);
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      fwd = RnnPrimitive::Create<vanilla_rnn_forward>(prop,
          mode == rnn_enum::kRnnTanh ? algorithm::eltwise_tanh : algorithm::eltwise_relu,
          mkldnn_rnn_direction, src_layer_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc);
      break;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      break;
  }
  return fwd;
}

RnnBwdPrimitive GetRnnBwdPrim(const MKLDNNRnnForwardTraining &fwd,
                              const NDArray &data, const NDArray &params) {
  using namespace mkldnn;
  using tag = mkldnn::memory::format_tag;
  const MKLDNNRnnLayerParam& layer_param = fwd.GetParam();
  const int mode = layer_param.mode;
  memory::data_type data_type = get_mkldnn_type(data.dtype());
  memory::data_type weight_type = get_mkldnn_type(params.dtype());
  const prop_kind prop = prop_kind::backward;
  rnn_direction mkldnn_rnn_direction = layer_param.bidirectional ?
      rnn_direction::bidirectional_concat : rnn_direction::unidirectional;

  auto src_layer_desc    = memory::desc(layer_param.src_dims, data_type, tag::tnc);
  auto weight_layer_desc = memory::desc(layer_param.weight_layer_dims, weight_type, tag::any);
  auto weight_iter_desc  = memory::desc(layer_param.weight_iter_dims, weight_type, tag::any);
  auto bias_desc         = memory::desc(layer_param.bias_dims, data_type, tag::ldgo);
  auto dst_layer_desc    = memory::desc(layer_param.dst_dims, data_type, tag::tnc);
  auto src_state_desc    = memory::desc(layer_param.state_dims, data_type, tag::ldnc);
  auto dst_state_desc = layer_param.state_outputs ? memory::desc(
      layer_param.state_dims, data_type, tag::ldnc) : memory::desc();

  const void* fwd_pd = fwd.GetPrimDesc();
  auto bwd = RnnBwdPrimitive();
  switch (mode) {
    case rnn_enum::kLstm: {
      const lstm_forward::primitive_desc* pd =
          reinterpret_cast<const lstm_forward::primitive_desc*>(fwd_pd);
      bwd = RnnBwdPrimitive::Create<lstm_forward, lstm_backward>(*pd,
          prop, mkldnn_rnn_direction,
          // data desc
          src_layer_desc, src_state_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc,
          dst_state_desc,
          // diff desc
          src_layer_desc, src_state_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc,
          dst_state_desc);
    } break;
    case rnn_enum::kGru: {
      const lbr_gru_forward::primitive_desc* pd =
          reinterpret_cast<const lbr_gru_forward::primitive_desc*>(fwd_pd);
      bwd = RnnBwdPrimitive::Create<lbr_gru_forward, lbr_gru_backward>(*pd,
          prop, mkldnn_rnn_direction,
          // data desc
          src_layer_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc,
          // diff desc
          src_layer_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc);
    } break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh: {
      const vanilla_rnn_forward::primitive_desc* pd =
          reinterpret_cast<const vanilla_rnn_forward::primitive_desc*>(fwd_pd);
      bwd = RnnBwdPrimitive::Create<vanilla_rnn_forward, vanilla_rnn_backward>(
          *pd, prop,
          mode == rnn_enum::kRnnTanh ? algorithm::eltwise_tanh : algorithm::eltwise_relu,
          mkldnn_rnn_direction,
          // data desc
          src_layer_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc,
          // diff desc
          src_layer_desc, src_state_desc, weight_layer_desc,
          weight_iter_desc, bias_desc, dst_layer_desc, dst_state_desc);
    } break;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      break;
  }
  return bwd;
}

/*
 * Naive weights layout is:
 *         | l0_l2r_wx | l0_l2r_wh | l0_r2l_wx | l0_r2l_wh |
 *         | l1_l2r_wx | l1_l2r_wh | l1_r2l_wx | l1_r2l_wh |
 *         ...
 *
 * We need concat them to be:
 *         | l0_l2r_wx | l0_r2l_wx | l1_l2r_wx | l1_r2l_wx |
 *         | l0_l2r_wh | l0_r2l_wh | l1_l2r_wh | l1_r2l_wh |
 *         ...
 *
 * All the memory blocks are in goi format.
 */
static void ConcatWeights(const mkldnn::memory &dst,
                          const int concat_dimension,
                          const std::vector<void*> &src_ptrs,
                          const mkldnn::memory::format_tag src_format) {
  using memory = mkldnn::memory;
  auto cpu_engine = dst.get_engine();
  mkldnn::stream s(cpu_engine);
  const memory::desc& dst_desc = dst.get_desc();
  // Use dst memory dims to initialize src memory dims, then set the concat
  // dim to 1. And Rnn weights are 5-dimension tensor.
  memory::dims src_dims(dst_desc.data.dims, dst_desc.data.dims + 5);
  src_dims.at(concat_dimension) = 1;
  std::vector<memory::desc> src_descs;
  std::unordered_map<int, memory> concat_args;

  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    src_descs.emplace_back(src_dims,
        static_cast<memory::data_type>(dst_desc.data.data_type), src_format);
    concat_args.emplace(MKLDNN_ARG_MULTIPLE_SRC + i,
        memory(src_descs.back(), cpu_engine, src_ptrs.at(i)));
  }
  concat_args.emplace(MKLDNN_ARG_DST, dst);

  auto concat_pd = mkldnn::concat::primitive_desc(dst.get_desc(),
      concat_dimension, src_descs, cpu_engine);
  mkldnn::concat(concat_pd).execute(s, concat_args);
}

#define RNN_HANDLE_FUNC_NAME set_handle
#define RNN_HANDLE_FUNC(RNN_FUNC_NAME)                                         \
auto RNN_FUNC_NAME = [&cpu_engine, &args](int arg_name, const desc& md,        \
    void* handle) {                                                            \
  if (args.find(arg_name) != args.end()) {                                     \
    if (handle != nullptr) args.at(arg_name).set_data_handle(handle);          \
  } else {                                                                     \
    args[arg_name] = handle ? mkldnn::memory(md, cpu_engine, handle)           \
        : mkldnn::memory(md, cpu_engine);                                      \
  }                                                                            \
}

#define RNN_FWD_SET(NAME, DIMS, TAG, HANDLE, DTYPE) \
RNN_FWD_SET_(RNN_HANDLE_FUNC_NAME, NAME, DIMS, TAG, HANDLE, DTYPE)

#define RNN_FWD_SET_(FUNC, NAME, DIMS, TAG, HANDLE, DTYPE) \
FUNC(MKLDNN_ARG_##NAME, {DIMS, get_mkldnn_type(DTYPE), TAG}, HANDLE)

#define RNN_BWD_SET(NAME, ARGS, HANDLE) \
RNN_BWD_SET_(RNN_HANDLE_FUNC_NAME, NAME, ARGS, HANDLE)

#define RNN_BWD_SET_(FUNC, NAME, ARGS, HANDLE) \
FUNC(MKLDNN_ARG_DIFF_##NAME, ARGS.at(MKLDNN_ARG_##NAME).get_desc(), HANDLE)

/*
 * Set new src data handler to Forward memory. The memory primitives are
 * not initialized until SetNewDataMem is first invoked. Src data handler
 * must not be nullptr, except for cx with LSTM mode. If either hy, cy is
 * nullptr, it may run with non-state_ouput or non-LSTM mode. Thus, the
 * corresponding memory should be a empty mkldnn::memory().
 */
void MKLDNNRnnForward::SetNewDataMem(void* x, void* hx, void* cx,
                                     void* y, void* hy, void* cy,
                                     const int dtype) {
  using dims = mkldnn::memory::dims;
  using desc = mkldnn::memory::desc;
  using format_tag = mkldnn::memory::format_tag;
  auto& cpu_engine = CpuEngine::Get()->get_engine();
  mkldnn_args_map_t& args = net_args_;

  RNN_HANDLE_FUNC(RNN_HANDLE_FUNC_NAME);

  // Set various data memory
  RNN_FWD_SET(SRC,      param_.src_dims,   format_tag::tnc,  x,  dtype);
  RNN_FWD_SET(DST,      param_.dst_dims,   format_tag::tnc,  y,  dtype);
  RNN_FWD_SET(SRC_ITER, param_.state_dims, format_tag::ldnc, hx, dtype);

  if (param_.state_outputs) {
    RNN_FWD_SET(DST_ITER, param_.state_dims, format_tag::ldnc, hy, dtype);
  }

  if (param_.mode == rnn_enum::kLstm) {
    RNN_FWD_SET(SRC_ITER_C, param_.state_dims, format_tag::ldnc, cx, dtype);
    if (param_.state_outputs) {
      RNN_FWD_SET(DST_ITER_C, param_.state_dims, format_tag::ldnc, cy, dtype);
    }
  }
}

/*
 * Reorder the concatenated weights memory to a efficient memory block
 * with primitive-prefered format.
 */
void MKLDNNRnnForward::ReorderWeights() {
  auto& cpu_engine = CpuEngine::Get()->get_engine();
  mkldnn::stream s(cpu_engine);
  mkldnn::reorder(*weights_layer_r_, *weights_layer_)
      .execute(s, *weights_layer_r_, *weights_layer_);
  mkldnn::reorder(*weights_iter_r_, *weights_iter_)
      .execute(s, *weights_iter_r_, *weights_iter_);
  s.wait();
}

void AdjustGruGateOrder(char* weight,
                        const size_t input_size,
                        const size_t hidden_size,
                        const int dtype) {
  // mxnet gru gate order is reset, update and new gates
  // mkldnn gru gate order is update, reset and new gates
  size_t single_weight_bytes = input_size * hidden_size * mshadow::mshadow_sizeof(dtype);
  char* weight_reset = weight;
  char* weight_update = weight + single_weight_bytes;
  std::swap_ranges(weight_reset, weight_update, weight_update);
}

/*
 * Fuse uni-directional bias among single layer.
 */
template <typename DType>
void FuseBias(DType* fuse_bias, DType* naive_bias,
              const int mode, const size_t state_size) {
  const size_t ngates = GetRnnGatesNum(mode);
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const size_t nbias = mode == rnn_enum::kGru ? ngates + 1 : ngates;
  // MSVC-14.0 (OpenMP 2.0 compatible) doesn't support unsigned integral type in
  // OpenMP 'for' statement.
  const int state_size_ = static_cast<int>(state_size);
  const int single_b_sz = static_cast<int>(nbias * state_size);
  DType* bx = naive_bias;
  DType* bh = naive_bias + state_size * ngates;
  if (mode == rnn_enum::kGru) {
    // While mxnet gru gate order is reset, update and new gates,
    // mkldnn gru gate order is update, reset and new gates. So
    // we need to swap the order of reset and update from mxnet.
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < state_size_; j++) {
      // Swap summed reset, update bias
      fuse_bias[j + state_size] = bx[j] + bh[j];
      fuse_bias[j] = bx[j + state_size] + bh[j + state_size];

      // Memcpy two new gates
      fuse_bias[j + 2 * state_size] = bx[j + 2 * state_size];
      fuse_bias[j + 3 * state_size] = bh[j + 2 * state_size];
    }
  } else {
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < single_b_sz; ++j) {
      // Sum two bias
      fuse_bias[j] = bx[j] + bh[j];
    }
  }
}

inline void EmplaceNetArgs(mkldnn_args_map_t* net_args, const int arg_name,
                           const mkldnn::memory* mem) {
  if (net_args->find(arg_name) != net_args->end()) {
    if (net_args->at(arg_name).get_data_handle() == mem->get_data_handle()) {
      return;
    } else {
      net_args->at(arg_name).set_data_handle(mem->get_data_handle());
    }
  } else {
    net_args->emplace(arg_name, *mem);
  }
}

/*
 * Copy naive memory to mkldnn-format memory. It will initialize the memory
 * when first invoked. Then, the naive weight_layer and weight_iter are
 * concatenated to xxx_xx_r memory. Per the different gates order of GRU,
 * it will swap the memory blocks of gates among concatenated memory
 * inplace. From then on, the xxx_xx_r memory is reordered to target
 * memory with preferred format_tag. Finally, naive bias is fused to MKLDNN
 * bias memory.
 */
void MKLDNNRnnForward::SetWeightsMem(MKLDNNRnnMemMgr* mgr, void *w_ptr, void *b_ptr,
                                     const bool is_train, const int dtype) {
  using format_tag = mkldnn::memory::format_tag;
  auto mkldnn_dtype = get_mkldnn_type(dtype);
  // Get the weights' memory for RNN forward primitive
  if (weights_layer_ == nullptr) {
    weights_layer_ = mgr->Alloc(fwd_inf_.GetLayerDesc());
  }
  if (weights_iter_ == nullptr) {
    weights_iter_ = mgr->Alloc(fwd_inf_.GetIterDesc());
  }
  if (bias_ == nullptr) {
    bias_ = mgr->Alloc(
        {param_.bias_dims, mkldnn_dtype, format_tag::ldgo});
  }

  // Get the intermediate memory for weights concat & reorder
  if (weights_layer_r_ == nullptr) {
    weights_layer_r_ = mgr->Alloc(
        {param_.weight_layer_dims, mkldnn_dtype, format_tag::ldgoi});
  }
  if (weights_iter_r_ == nullptr) {
    weights_iter_r_ = mgr->Alloc(
        {param_.weight_iter_dims, mkldnn_dtype, format_tag::ldgoi});
  }

  // Get the bytes of a real type
  size_t dtype_bytes = mshadow::mshadow_sizeof(dtype);

  // convert void* to char* for arithmetic operations
  char *weights_ptr = static_cast<char *>(w_ptr);
  size_t wx_bytes = GetRnnGatesNum(param_.mode) * param_.state_size *
        param_.input_size * dtype_bytes;  //* DIMS: ngates x state_size x input_size
  size_t wh_bytes = GetRnnGatesNum(param_.mode) * param_.state_size *
        param_.state_size * dtype_bytes;  //* DIMS: ngates x state_size x state_size
  char *l2r_wx = weights_ptr;
  char *l2r_wh = l2r_wx + wx_bytes;       //* DIMS: ngates x state_size * state_size

  if (param_.num_layer == 1 && param_.bidirectional) {
    //* single bidirectinal layer, concat weights on direction axis
    char *r2l_wx = weights_ptr + param_.single_w_size * dtype_bytes;
    char *r2l_wh = r2l_wx + wx_bytes;  //* DIMS: ngates x state_size * state_size
    ConcatWeights(*weights_layer_r_, 1, {l2r_wx, r2l_wx}, format_tag::ldgoi);
    ConcatWeights(*weights_iter_r_, 1, {l2r_wh, r2l_wh}, format_tag::ldgoi);
  } else if (param_.num_layer == 1 && !param_.bidirectional) {
    //* single uni-directional layer, no concatenate operator needed
    std::memcpy(weights_layer_r_->get_data_handle(), l2r_wx, wx_bytes);
    std::memcpy(weights_iter_r_->get_data_handle(), l2r_wh, wh_bytes);
  } else if (param_.num_layer > 1 && !param_.bidirectional) {
    //* concat fused multi-layer weights on layer axis
    std::vector<void *> l2r_wx_ptrs;
    std::vector<void *> l2r_wh_ptrs;
    for (int lyr = 0; lyr < param_.num_layer; ++lyr) {
      char *lth_wx = l2r_wx + lyr * param_.single_w_size * dtype_bytes;
      char *lth_wh = lth_wx + wx_bytes;
      l2r_wx_ptrs.push_back(lth_wx);
      l2r_wh_ptrs.push_back(lth_wh);
    }
    ConcatWeights(*weights_layer_r_, 0, l2r_wx_ptrs, format_tag::ldgoi);
    ConcatWeights(*weights_iter_r_, 0, l2r_wh_ptrs, format_tag::ldgoi);
  } else {
    LOG(FATAL) << "Undifined RNN fusion workflow for num_layer = " << param_.num_layer
               << ", and bidirectional is " << param_.bidirectional;
  }

  // Adjust gates order of LBR-GRU among concatenated memory inplace.
  char* fused_wx = static_cast<char*>(weights_layer_r_->get_data_handle());
  char* fused_wh = static_cast<char*>(weights_iter_r_->get_data_handle());
  if (param_.mode == rnn_enum::kGru) {
    for (size_t lyr = 0; lyr < static_cast<size_t>(param_.num_layer); ++lyr) {
      for (size_t d = 0; d < param_.bidirectional + 1U; ++d) {
        AdjustGruGateOrder(fused_wx, param_.input_size, param_.state_size, dtype);
        AdjustGruGateOrder(fused_wh, param_.state_size, param_.state_size, dtype);
        fused_wx += wx_bytes;
        fused_wh += wh_bytes;
      }
    }
  }
  // Reorder after adjustment only when is_train == false. When is_train == true, i.e.
  // in forward training path, we use plain memory (ldxxx) as the space for weights and
  // their gradients. Then, forward training primitives could fetch them from the scope
  // of forward inference. And from there, we don't need to reorder the plain memory to
  // the optimal rnn-packed memory for forward inference.
  ReorderWeights();

  // Process bias
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    DType* naive_b_ptr = static_cast<DType*>(b_ptr);
    DType* fused_bias = static_cast<DType*>(bias_->get_data_handle());
    for (int lyr = 0; lyr < param_.num_layer; ++lyr) {
      for (int d = 0; d < param_.bidirectional + 1; ++d) {
        FuseBias<DType>(fused_bias, naive_b_ptr, param_.mode, param_.state_size);
        fused_bias += param_.single_b_size;
        naive_b_ptr += param_.naive_single_b_size;
      }
    }
  });

  // insert weights into net_args
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_WEIGHTS_LAYER, this->weights_layer_);
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_WEIGHTS_ITER,  this->weights_iter_);
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_BIAS,          this->bias_);

  initialized_ = true;
}

void MKLDNNRnnForwardTraining::SetTrnMem(const MKLDNNRnnForward& fwd) {
  using memory = mkldnn::memory;
  const auto& cpu_engine = CpuEngine::Get()->get_engine();
  auto s = mkldnn::stream(cpu_engine);
  // Prepare mkldnn::memorys for weights_layer, weight_iter, and workspace
  if (workspace_ == nullptr)
    workspace_ = mkldnn_shared_mem_t(new memory(fwd_trn_.GetWorkspaceDesc(), cpu_engine));
  if (weights_layer_ == nullptr)
    weights_layer_ = mkldnn_shared_mem_t(new memory(fwd_trn_.GetLayerDesc(), cpu_engine));
  if (weights_iter_ == nullptr)
    weights_iter_ = mkldnn_shared_mem_t(new memory(fwd_trn_.GetIterDesc(), cpu_engine));

  // fill weights memory using the reordered weights of fwd_inference primitive
  if (fwd.weights_layer_r_->get_desc() == fwd_trn_.GetLayerDesc()) {
    weights_layer_->set_data_handle(fwd.weights_layer_r_->get_data_handle());
  } else {
    mkldnn::reorder(*fwd.weights_layer_r_, *weights_layer_)
        .execute(s, *fwd.weights_layer_r_, *weights_layer_);
  }

  if (fwd.weights_iter_r_->get_desc() == fwd_trn_.GetIterDesc()) {
    weights_iter_->set_data_handle(fwd.weights_iter_r_->get_data_handle());
  } else {
    mkldnn::reorder(*fwd.weights_iter_r_, *weights_iter_)
        .execute(s, *fwd.weights_iter_r_, *weights_iter_);
  }
  s.wait();

  // bias are always in format_tag::ldgo
  this->bias_ = fwd.bias_;

  // insert weights into net_args
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_WEIGHTS_LAYER, this->weights_layer_.get());
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_WEIGHTS_ITER,  this->weights_iter_.get());
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_BIAS,          this->bias_);
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_WORKSPACE,     this->workspace_.get());
}

void MKLDNNRnnForwardTraining::FetchData(const MKLDNNRnnForward& fwd) {
  for (auto& kv : fwd.net_args_) {
    switch (kv.first) {
      case MKLDNN_ARG_WEIGHTS_LAYER:
      case MKLDNN_ARG_WEIGHTS_ITER:
      case MKLDNN_ARG_BIAS:
      case MKLDNN_ARG_WORKSPACE:
        continue;

      default:
        EmplaceNetArgs(&this->net_args_, kv.first, &kv.second);
    }
  }
}

void MKLDNNRnnOp::Init(const OpContext &ctx,
                       const std::vector<NDArray> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<NDArray> &outputs) {
  using memory = mkldnn::memory;
  using format_tag = mkldnn::memory::format_tag;

  // In the `autograd.record()` context, RNNOp is required to run into
  // `forward_training` mode.
  const bool is_training = (ctx.is_train || ctx.need_grad);
  const size_t num_fusion = full_param_.layer_params.size();
  if (fwd_inf_vec_.size() < num_fusion) {
    size_t buffer_size = 0;  // Element number, instead of bytes, in the buffer
    for (auto& layer_param : full_param_.layer_params) {
      buffer_size += layer_param.workspace_size + layer_param.reserve_size;
    }
    buffer_size += outputs[rnn_enum::kOut].data().Size() * (num_fusion - 1);
    buffer_size += kMKLDNNAlign * num_fusion * 5;  // Add margin for alignment

    for (auto& layer_param : full_param_.layer_params) {
      fwd_inf_vec_.emplace_back(layer_param,
          ctx.is_train, inputs[rnn_enum::kData], inputs[rnn_enum::kParams]);
      buffer_size += fwd_inf_vec_.back().GetSize(inputs[rnn_enum::kParams].dtype());
    }
    mgr_.Init(buffer_size, ctx.run_ctx.ctx, inputs[rnn_enum::kParams].dtype());
  }

  if (is_training && fwd_trn_vec_.size() < num_fusion) {
    for (auto& layer_param : full_param_.layer_params) {
      fwd_trn_vec_.emplace_back(layer_param,
          true, inputs[rnn_enum::kData], inputs[rnn_enum::kParams]);
    }
  }

  // Get the bytes of a real type
  const NDArray &weights = inputs[rnn_enum::kParams];
  int dtype = weights.dtype();
  size_t dtype_bytes = mshadow::mshadow_sizeof(dtype);

  const RNNParam &default_param = full_param_.default_param;
  char *weights_ptr = static_cast<char *>(weights.data().dptr_);
  char *bias_ptr = weights_ptr + (weights.data().Size() -
      GetRnnBiasSize(default_param.num_layers, default_param.state_size,
        default_param.bidirectional + 1, default_param.mode)) * dtype_bytes;
  for (auto& fwd_layer : fwd_inf_vec_) {
    size_t single_w_bytes = fwd_layer.GetParam().single_w_size * dtype_bytes;
    size_t single_b_bytes = fwd_layer.GetParam().naive_single_b_size * dtype_bytes;
    size_t directions = fwd_layer.GetParam().bidirectional ? 2 : 1;
    size_t layer_weights_bytes = single_w_bytes * directions;
    size_t layer_bias_bytes = single_b_bytes * directions;  // Naive MXNet has double bias

    if (!fwd_layer.IsInitialized() || is_training)
      fwd_layer.SetWeightsMem(&(this->mgr_), weights_ptr, bias_ptr, is_training, dtype);
    weights_ptr += layer_weights_bytes;
    bias_ptr += layer_bias_bytes;
  }

  if (is_training) {
    CHECK_EQ(fwd_trn_vec_.size(), fwd_inf_vec_.size()) <<
      "Layers' configurations of forward inference and forward training are disparate.";
    for (size_t lyr = 0; lyr < fwd_inf_vec_.size(); ++lyr)
      fwd_trn_vec_.at(lyr).SetTrnMem(fwd_inf_vec_.at(lyr));
  }

  CHECK_EQ(num_fusion, fwd_inf_vec_.size()) <<
      "Layer vector's size has a different value than the number of fusion.";
  if (dst_.size() < num_fusion - 1) {
    int data_dtype = outputs[rnn_enum::kOut].dtype();
    // Here we need `fwd_inf_vec_.size() - 1` spaces for the intermediate results of the multiple
    // fused layers. And for the result of the last fused layer, `outputs[rnn_enum::kOut]` could
    // provide the space. Hence, `forward_inf_vec_.back()` is excluded when allocates the spaces
    // for intermediate results.
    for (std::vector<MKLDNNRnnForward>::const_iterator fwd = fwd_inf_vec_.begin();
        fwd != fwd_inf_vec_.end() - 1; ++fwd)
      dst_.push_back(mgr_.Alloc(
        {fwd->GetParam().dst_dims, get_mkldnn_type(data_dtype), format_tag::tnc}));
  }

  initialized_ = true;
}

void MKLDNNRnnBackward::FetchDataWeightsMem(const MKLDNNRnnForwardTraining& fwd) {
  using memory = mkldnn::memory;
  auto& cpu_engine = CpuEngine::Get()->get_engine();
  auto s = mkldnn::stream(cpu_engine);

  if (this->weights_layer_ == nullptr)
    this->weights_layer_ = mkldnn_shared_mem_t(new memory(bwd_.weights_layer_desc_, cpu_engine));
  if (this->weights_iter_ == nullptr)
    this->weights_iter_ = mkldnn_shared_mem_t(new memory(bwd_.weights_iter_desc_, cpu_engine));

  for (auto& kv : fwd.net_args_) {
    const mkldnn::memory* valid_mem;
    switch (kv.first) {
      case MKLDNN_ARG_WEIGHTS_LAYER: {
        if (bwd_.weights_layer_desc_ == fwd.fwd_trn_.GetLayerDesc()) {
          this->weights_layer_->set_data_handle(kv.second.get_data_handle());
        } else {
          mkldnn::reorder(*fwd.weights_layer_, *this->weights_layer_)
              .execute(s, *fwd.weights_layer_, *this->weights_layer_);
        }
        valid_mem = this->weights_layer_.get();
      } break;
      case MKLDNN_ARG_WEIGHTS_ITER: {
        if (bwd_.weights_iter_desc_ == fwd.fwd_trn_.GetLayerDesc()) {
          this->weights_iter_->set_data_handle(kv.second.get_data_handle());
        } else {
          mkldnn::reorder(*fwd.weights_iter_, *this->weights_iter_)
              .execute(s, *fwd.weights_iter_, *this->weights_iter_);
        }
        valid_mem = this->weights_iter_.get();
      } break;

      default:
        valid_mem = &kv.second;
    }
    EmplaceNetArgs(&this->net_args_, kv.first, valid_mem);
  }
  s.wait();
}

void MKLDNNRnnBackward::SetWeightsGradsMem() {
  auto& cpu_engine = CpuEngine::Get()->get_engine();
  if (this->diff_weights_layer_ == nullptr)
    this->diff_weights_layer_ = std::make_shared<mkldnn::memory>(
        bwd_.diff_weights_layer_desc_, cpu_engine);
  if (this->diff_weights_iter_ == nullptr)
    this->diff_weights_iter_ = std::make_shared<mkldnn::memory>(
        bwd_.diff_weights_iter_desc_, cpu_engine);
  if (this->diff_bias_ == nullptr)
    this->diff_bias_ = std::make_shared<mkldnn::memory>(
        bwd_.diff_bias_desc_, cpu_engine);
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_DIFF_WEIGHTS_LAYER,
      this->diff_weights_layer_.get());
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_DIFF_WEIGHTS_ITER,
      this->diff_weights_iter_.get());
  EmplaceNetArgs(&this->net_args_, MKLDNN_ARG_DIFF_BIAS,
      this->diff_bias_.get());
}

void MKLDNNRnnBackward::SetDataGradsMem(
    void* diff_src, void* diff_state, void* diff_statecell,
    void* diff_dst, void* diff_state_out, void* diff_statecell_out,
    const int dtype) {
  using desc = mkldnn::memory::desc;
  auto& cpu_engine = CpuEngine::Get()->get_engine();
  mkldnn_args_map_t& args = this->net_args_;

  RNN_HANDLE_FUNC(RNN_HANDLE_FUNC_NAME);

  // Set various diff memory
  auto& fwd_args = fwd_ptr_->GetArgsMap();
  RNN_BWD_SET(SRC,      fwd_args, diff_src);
  RNN_BWD_SET(SRC_ITER, fwd_args, diff_state);
  RNN_BWD_SET(DST,      fwd_args, diff_dst);

  if (fwd_ptr_->GetParam().state_outputs)
    RNN_BWD_SET(DST_ITER, fwd_args, diff_state_out);

  if (fwd_ptr_->GetParam().mode == rnn_enum::kLstm) {
    RNN_BWD_SET(SRC_ITER_C, fwd_args, diff_statecell);
    if (fwd_ptr_->GetParam().state_outputs) {
      RNN_BWD_SET(DST_ITER_C, fwd_args, diff_statecell_out);
    }
  }
}

void MKLDNNRnnBackward::CommitWeightsDiff(void* diff_weights, void* diff_bias,
                                          const OpReqType req, const int dtype) {
  using tag = mkldnn::memory::format_tag;
  auto& cpu_engine = CpuEngine::Get()->get_engine();
  auto s = mkldnn::stream(cpu_engine);

  const MKLDNNRnnLayerParam& param = fwd_ptr_->GetParam();
  const int num_layer = param.num_layer;
  const int direction = param.bidirectional ? 2 : 1;
  const int ngates = GetRnnGatesNum(param.mode);
  const size_t dtype_bytes = mshadow::mshadow_sizeof(dtype);
  const size_t wxh_size = param.single_w_size;
  const size_t wx_size = param.input_size * param.state_size * ngates;
  const size_t wh_size = param.state_size * param.state_size * ngates;
  const size_t wxh_bytes = param.single_w_size * dtype_bytes;
  const size_t wx_bytes = param.input_size * param.state_size * ngates * dtype_bytes;
  const size_t wh_bytes = param.state_size * param.state_size * ngates * dtype_bytes;

  /* naive weights layout is:
          1st-layer: | wx_lr  | wh_lr  | wx_rl | wh_rl |
          2st-layer: | wx_lr  | wh_lr  | wx_rl | wh_rl |
  size:              |    wxh_bytes    |
                     |wx_bytes|wh_bytes|      
  */
  if (kWriteTo == req) {
    char* naive_weights = static_cast<char *>(diff_weights);
    char* diff_wx_ptr = static_cast<char *>(diff_weights_layer_->get_data_handle());
    char* diff_wh_ptr = static_cast<char *>(diff_weights_iter_->get_data_handle());
    if (param.mode != rnn_enum::kGru) {
      for (int shift = 0; shift < num_layer * direction; ++shift) {
        std::memcpy(naive_weights + shift * wxh_bytes,
            diff_wx_ptr + shift * wx_bytes, wx_bytes);
      }
      // align naive_weights to weights_iter memory
      naive_weights += wx_bytes;
      for (int shift = 0; shift < num_layer * direction; ++shift) {
        std::memcpy(naive_weights + shift * wxh_bytes,
            diff_wh_ptr + shift * wh_bytes, wh_bytes);
      }
    } else {
      const size_t wx_bytes_per_gate = param.input_size * param.state_size * dtype_bytes;
      const size_t wh_bytes_per_gate = param.state_size * param.state_size * dtype_bytes;
      for (int shift = 0; shift < num_layer * direction; ++shift) {
        std::memcpy(naive_weights + shift * wxh_bytes + wx_bytes_per_gate,
            diff_wx_ptr + shift * wx_bytes, wx_bytes_per_gate);
        std::memcpy(naive_weights + shift * wxh_bytes,
            diff_wx_ptr + shift * wx_bytes + wx_bytes_per_gate, wx_bytes_per_gate);
        std::memcpy(naive_weights + shift * wxh_bytes + 2 * wx_bytes_per_gate,
            diff_wx_ptr + shift * wx_bytes + 2 * wx_bytes_per_gate, wx_bytes_per_gate);
      }
      // align naive_weights to weights_iter memory
      naive_weights += wx_bytes;
      for (int shift = 0; shift < num_layer * direction; ++shift) {
        std::memcpy(naive_weights + shift * wxh_bytes + wh_bytes_per_gate,
            diff_wh_ptr + shift * wh_bytes, wh_bytes_per_gate);
        std::memcpy(naive_weights + shift * wxh_bytes,
            diff_wh_ptr + shift * wh_bytes + wh_bytes_per_gate, wh_bytes_per_gate);
        std::memcpy(naive_weights + shift * wxh_bytes + 2 * wh_bytes_per_gate,
            diff_wh_ptr + shift * wh_bytes + 2 * wh_bytes_per_gate, wh_bytes_per_gate);
      }
    }
  } else if (kAddTo == req) {
    if (param.mode != rnn_enum::kGru) {
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        DType* naive_weights = static_cast<DType *>(diff_weights);
        DType* diff_wx_ptr = static_cast<DType *>(diff_weights_layer_->get_data_handle());
        DType* diff_wh_ptr = static_cast<DType *>(diff_weights_iter_->get_data_handle());
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          common::ParallelAdd(naive_weights + shift * wxh_size,
              diff_wx_ptr + shift * wx_size, wx_size);
        }
        // align naive_weights to weights_iter memory
        naive_weights += wx_size;
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          common::ParallelAdd(naive_weights + shift * wxh_size,
              diff_wh_ptr + shift * wh_size, wh_size);
        }
      });
    } else {
      const size_t wx_size_per_gate = param.input_size * param.state_size;
      const size_t wh_size_per_gate = param.state_size * param.state_size;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        DType* naive_weights = static_cast<DType *>(diff_weights);
        DType* diff_wx_ptr = static_cast<DType *>(diff_weights_layer_->get_data_handle());
        DType* diff_wh_ptr = static_cast<DType *>(diff_weights_iter_->get_data_handle());
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          common::ParallelAdd(naive_weights + shift * wxh_size + wx_size_per_gate,
              diff_wx_ptr + shift * wx_size, wx_size_per_gate);
          common::ParallelAdd(naive_weights + shift * wxh_size,
              diff_wx_ptr + shift * wx_size + wx_size_per_gate, wx_size_per_gate);
          common::ParallelAdd(naive_weights + shift * wxh_size + 2 * wx_size_per_gate,
              diff_wx_ptr + shift * wx_size + 2 * wx_size_per_gate, wx_size_per_gate);
        }
        // align naive_weights to weights_iter memory
        naive_weights += wx_size;
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          common::ParallelAdd(naive_weights + shift * wxh_size + wh_size_per_gate,
              diff_wh_ptr + shift * wh_size, wh_size_per_gate);
          common::ParallelAdd(naive_weights + shift * wxh_size,
              diff_wh_ptr + shift * wh_size + wh_size_per_gate, wh_size_per_gate);
          common::ParallelAdd(naive_weights + shift * wxh_size + 2 * wh_size_per_gate,
              diff_wh_ptr + shift * wh_size + 2 * wh_size_per_gate, wh_size_per_gate);
        }
      });
    }
  }

  if (kWriteTo == req) {
    const size_t bias_bytes = param.single_b_size * dtype_bytes;
    const size_t naive_bias_bytes = param.naive_single_b_size * dtype_bytes;
    char* naive_bias = static_cast<char *>(diff_bias);
    char* diff_bias_ptr = static_cast<char *>(this->diff_bias_->get_data_handle());
    if (param.mode != rnn_enum::kGru) {
      for (int shift = 0; shift < num_layer * direction; ++shift) {
        std::memcpy(naive_bias + shift * naive_bias_bytes,
            diff_bias_ptr + shift * bias_bytes, bias_bytes);
        std::memcpy(naive_bias + shift * naive_bias_bytes + bias_bytes,
            diff_bias_ptr + shift * bias_bytes, bias_bytes);
      }
    } else {
      const size_t bias_bytes_per_gate = param.state_size * dtype_bytes;
      for (int shift = 0; shift < num_layer * direction; ++shift) {
        char* naive_reset = naive_bias + shift * naive_bias_bytes;
        char* naive_update = naive_reset + bias_bytes_per_gate;
        char* update = diff_bias_ptr + shift * bias_bytes;
        char* reset = update + bias_bytes_per_gate;

        std::memcpy(naive_update, update, bias_bytes_per_gate);
        std::memcpy(naive_reset, reset, bias_bytes_per_gate);
        std::memcpy(naive_update + naive_bias_bytes / 2, update, bias_bytes_per_gate);
        std::memcpy(naive_reset + naive_bias_bytes / 2, reset, bias_bytes_per_gate);

        char* naive_new_bx = naive_update + bias_bytes_per_gate;
        char* naive_new_bh = naive_new_bx + naive_bias_bytes / 2;
        char* new_bx = reset + bias_bytes_per_gate;
        char* new_bh = new_bx + bias_bytes_per_gate;
        std::memcpy(naive_new_bx, new_bx, bias_bytes_per_gate);
        std::memcpy(naive_new_bh, new_bh, bias_bytes_per_gate);
      }
    }
  } else if (kAddTo == req) {
    const size_t bias_size = param.single_b_size;
    const size_t naive_bias_size = param.naive_single_b_size;
    if (param.mode != rnn_enum::kGru) {
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        DType* naive_bias = static_cast<DType *>(diff_bias);
        DType* diff_bias_ptr = static_cast<DType *>(this->diff_bias_->get_data_handle());
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          common::ParallelAdd(naive_bias + shift * naive_bias_size,
              diff_bias_ptr + shift * bias_size, bias_size);
          common::ParallelAdd(naive_bias + shift * naive_bias_size + bias_size,
              diff_bias_ptr + shift * bias_size, bias_size);
        }
      });
    } else {
      const size_t bias_size_per_gate = param.state_size;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        DType* naive_bias = static_cast<DType *>(diff_bias);
        DType* diff_bias_ptr = static_cast<DType *>(this->diff_bias_->get_data_handle());
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          DType* naive_reset = naive_bias + shift * naive_bias_size;
          DType* naive_update = naive_reset + bias_size_per_gate;
          DType* update = diff_bias_ptr + shift * bias_size;
          DType* reset = update + bias_size_per_gate;

          common::ParallelAdd(naive_update, update, bias_size_per_gate);
          common::ParallelAdd(naive_reset, reset, bias_size_per_gate);
          common::ParallelAdd(naive_update + naive_bias_size / 2, update, bias_size_per_gate);
          common::ParallelAdd(naive_reset + naive_bias_size / 2, reset, bias_size_per_gate);

          DType* naive_new_bx = naive_update + bias_size_per_gate;
          DType* naive_new_bh = naive_new_bx + naive_bias_size / 2;
          DType* new_bx = reset + bias_size_per_gate;
          DType* new_bh = new_bx + bias_size_per_gate;
          common::ParallelAdd(naive_new_bx, new_bx, bias_size_per_gate);
          common::ParallelAdd(naive_new_bh, new_bh, bias_size_per_gate);
        }
      });
    }
  }
}

template <typename MKLDNNRnnX>
inline void RegisterMKLDNNRnn(MKLDNNRnnX const& rnn) {
  MKLDNNStream::Get()->RegisterPrimArgs(rnn.GetFwd(), rnn.GetArgsMap());
}

template <>
inline void RegisterMKLDNNRnn(MKLDNNRnnBackward const& rnn) {
  MKLDNNStream::Get()->RegisterPrimArgs(rnn.GetBwd(), rnn.GetArgsMap());
}

void MKLDNNRnnOp::Forward(const OpContext &ctx,
                          const std::vector<NDArray> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  // In the `autograd.record()` context, RNNOp is required to run into
  // forward_training mode.
  const bool is_training = (ctx.is_train || ctx.need_grad);
  const RNNParam& default_param = full_param_.default_param;

  // Initialize weights version
  if (!initialized_ && weights_version_ == 0) {
    weights_version_ = inputs[rnn_enum::kParams].version();
  }

  // Check if weights NDArray was changed. If so, reset initialized_
  if (weights_version_ != inputs[rnn_enum::kParams].version() &&
      fwd_inf_vec_.size() > 0) {
    initialized_ = false;
    for (auto& fwd : fwd_inf_vec_) fwd.Reset();
    weights_version_ = inputs[rnn_enum::kParams].version();
  }

  if (!initialized_ || is_training || fwd_inf_vec_.size() == 0) {
    Init(ctx, inputs, req, outputs);
  }

  // Get data type
  int data_dtype = inputs[rnn_enum::kData].dtype();
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

  if (fwd_inf_vec_.size() == 1) {
    fwd_inf_vec_.front().SetNewDataMem(src, src_state, src_state_cell,
        dst, dst_state, dst_state_cell, data_dtype);
    if (is_training) {
      fwd_trn_vec_.front().FetchData(fwd_inf_vec_.front());
    }
  } else {
    CHECK_EQ(fwd_inf_vec_.size(), dst_.size() + 1) << "Output memory error.";
    size_t cell_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ *
        default_param.state_size * mshadow::mshadow_sizeof(data_dtype);

    // Set input data memory for the first layer. This stores intermediate output
    // results in this->xxx, used as the source input of the next layer.
    fwd_inf_vec_.front().SetNewDataMem(src, src_state, src_state_cell,
        this->dst_.front()->get_data_handle(), dst_state, dst_state_cell, data_dtype);
    if (is_training) {
      fwd_trn_vec_.front().FetchData(fwd_inf_vec_.front());
    }
    // 1st_lyr -> dst_handle -> next_lyr -> dst_handle -> next_lyr -> ...
    for (size_t lyr = 1; lyr < fwd_inf_vec_.size() - 1; ++lyr) {
      src_state += cell_bytes;
      if (src_state_cell) src_state_cell += cell_bytes;
      if (dst_state) dst_state += cell_bytes;
      if (dst_state_cell) dst_state_cell += cell_bytes;
      fwd_inf_vec_.at(lyr).SetNewDataMem(this->dst_.at(lyr - 1)->get_data_handle(),
          src_state, src_state_cell,
          this->dst_.at(lyr)->get_data_handle(), dst_state, dst_state_cell, data_dtype);
      if (is_training) {
        fwd_trn_vec_.at(lyr).FetchData(fwd_inf_vec_.at(lyr));
      }
    }
    // Set output data memory for the last layer.
    src_state += cell_bytes;
    if (src_state_cell) src_state_cell += cell_bytes;
    if (dst_state) dst_state += cell_bytes;
    if (dst_state_cell) dst_state_cell += cell_bytes;
    fwd_inf_vec_.back().SetNewDataMem(this->dst_.back()->get_data_handle(),
        src_state, src_state_cell, dst, dst_state, dst_state_cell, data_dtype);
    if (is_training) {
      fwd_trn_vec_.back().FetchData(fwd_inf_vec_.back());
    }
  }
  if (is_training) {
    for (auto& trn_lyr : fwd_trn_vec_) RegisterMKLDNNRnn(trn_lyr);
  } else {
    for (auto& inf_lyr : fwd_inf_vec_) RegisterMKLDNNRnn(inf_lyr);
  }
  CommitOutput(outputs[rnn_enum::kOut], out_mem);
  if (default_param.state_outputs) {
    CommitOutput(outputs[rnn_enum::kStateOut], stateout_mem);
    if (default_param.mode == rnn_enum::kLstm)
      CommitOutput(outputs[rnn_enum::kStateCellOut], statecellout_mem);
  }
  MKLDNNStream::Get()->Submit();
}

void MKLDNNRnnOp::Backward(const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  using tag = mkldnn::memory::format_tag;
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  const RNNParam& default_param = full_param_.default_param;

  // Initialize the bwd_vec_
  if (bwd_vec_.size() != fwd_inf_vec_.size()) {
    bwd_vec_.clear();
    for (size_t lyr = 0; lyr < fwd_inf_vec_.size(); ++lyr)
      bwd_vec_.emplace_back(fwd_trn_vec_.at(lyr), inputs[rnn_enum::kData],
          inputs[rnn_enum::kParams]);
  }
  // Fetch weights, src and dst from Forward layer
  if (bwd_vec_.size() != fwd_trn_vec_.size())
    LOG(FATAL) << "MKL-DNN RNN fusion error.";
  for (size_t lyr = 0; lyr < bwd_vec_.size(); ++lyr) {
    bwd_vec_.at(lyr).FetchDataWeightsMem(fwd_trn_vec_.at(lyr));
    bwd_vec_.at(lyr).SetWeightsGradsMem();
  }

  const int data_dtype = inputs[rnn_enum::kData].dtype();
  const int w_dtype = inputs[rnn_enum::kParams].dtype();
  const size_t w_bytes = mshadow::mshadow_sizeof(w_dtype);
  // Get temporary memory for diff_src, diff_state, diff_statecell
  const int num_layers = default_param.num_layers;
  const int seq_length = default_param.seq_length_;
  const int batch_size = default_param.batch_size_;
  const int input_size = default_param.input_size_;
  const int state_size = default_param.state_size;
  const int directions = default_param.bidirectional ? 2 : 1;
  mkldnn::memory::desc src_desc({seq_length, batch_size, input_size},
      get_mkldnn_type(data_dtype), tag::tnc);
  mkldnn::memory::desc state_desc({num_layers, directions, batch_size, state_size},
      get_mkldnn_type(data_dtype), tag::ldnc);
  auto diff_input_mem = CreateMKLDNNMem(outputs[rnn_enum::kData], src_desc, req[rnn_enum::kData]);
  mkldnn_output_t diff_state_mem;
  mkldnn_output_t diff_statecell_mem;
  // index description of outputs NDArray
  //   0    1    2     3
  // | dx | dw | dhx | dcx|
  char* dx = static_cast<char *>(diff_input_mem.second->get_data_handle());
  char* dw = static_cast<char *>(outputs[rnn_enum::kParams].data().dptr_);
  char* db = dw + (inputs[rnn_enum::kParams].data().Size() -
      GetRnnBiasSize(default_param.num_layers, default_param.state_size,
        default_param.bidirectional + 1, default_param.mode)) * w_bytes;
  diff_state_mem = CreateMKLDNNMem(
      outputs[rnn_enum::kState], state_desc, req[rnn_enum::kState]);
  char* dhx = static_cast<char *>(diff_state_mem.second->get_data_handle());
  char* dcx = nullptr;
  if (full_param_.default_param.mode == rnn_enum::kLstm
      && req[rnn_enum::kStateCell] != kNullOp) {
    diff_statecell_mem = CreateMKLDNNMem(
        outputs[rnn_enum::kStateCell], state_desc, req[rnn_enum::kStateCell]);
    dcx = static_cast<char *>(diff_statecell_mem.second->get_data_handle());
  }

  // index description of inputs NDArray
  //   0   1   2    3   4    5     6    7    8     9
  // | x | w | hx | y | dy | hy | dhy | cx | cy | dcy |
  char* dy = static_cast<char *>(inputs[4].data().dptr_);
  char* dhy = nullptr;
  if (default_param.state_outputs)
    dhy = static_cast<char *>(inputs[6].data().dptr_);

  char* dcy = nullptr;
  if ((default_param.mode == rnn_enum::kLstm) && default_param.state_outputs)
    dcy = static_cast<char *>(inputs[9].data().dptr_);

  if (bwd_vec_.size() == 1) {
    bwd_vec_.back().SetDataGradsMem(dx, dhx, dcx, dy, dhy, dcy, data_dtype);
    RegisterMKLDNNRnn(bwd_vec_.back());
  } else {
    const size_t cell_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ *
        default_param.state_size * mshadow::mshadow_sizeof(data_dtype);
    if (diff_src == nullptr) {
      auto desc = mkldnn::memory::desc(full_param_.layer_params.back().src_dims,
          get_mkldnn_type(data_dtype), tag::tnc);
      diff_src = std::make_shared<mkldnn::memory>(desc, CpuEngine::Get()->get_engine());
    }
    // Sets primitives from bottom to top, then submits them in reversed order.
    bwd_vec_.front().SetDataGradsMem(dx, dhx, dcx,
        diff_src->get_data_handle(), dhy, dcy, data_dtype);
    for (size_t lyr = 1; lyr < bwd_vec_.size() - 1; ++lyr) {
      if (dhx) dhx += cell_bytes;
      if (dcx) dcx += cell_bytes;
      if (dhy) dhy += cell_bytes;
      if (dcy) dcy += cell_bytes;
      bwd_vec_.at(lyr).SetDataGradsMem(diff_src->get_data_handle(), dhx, dcx,
          diff_src->get_data_handle(), dhy, dcy, data_dtype);
    }
    if (dhx) dhx += cell_bytes;
    if (dcx) dcx += cell_bytes;
    if (dhy) dhy += cell_bytes;
    if (dcy) dcy += cell_bytes;
    bwd_vec_.back().SetDataGradsMem(diff_src->get_data_handle(), dhx, dcx,
        dy, dhy, dcy, data_dtype);

    for (std::vector<MKLDNNRnnBackward>::const_reverse_iterator bwd = bwd_vec_.rbegin();
        bwd != bwd_vec_.rend(); ++bwd) {
      RegisterMKLDNNRnn(*bwd);
    }
  }
  CommitOutput(outputs[rnn_enum::kData], diff_input_mem);
  CommitOutput(outputs[rnn_enum::kState], diff_state_mem);
  if (full_param_.default_param.mode == rnn_enum::kLstm)
    CommitOutput(outputs[rnn_enum::kStateCell], diff_statecell_mem);
  MKLDNNStream::Get()->Submit();

  // Commit weights diff
  if (req[rnn_enum::kParams] != kNullOp) {
    for (size_t lyr = 0; lyr < bwd_vec_.size(); ++lyr) {
      bwd_vec_.at(lyr).CommitWeightsDiff(dw, db, req[rnn_enum::kParams], w_dtype);
      dw += full_param_.layer_params.at(lyr).single_w_size * w_bytes;
      db += full_param_.layer_params.at(lyr).single_b_size * w_bytes;
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
