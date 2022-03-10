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
 * \file dnnl_rnn.cc
 * \brief Common functions used by DNNL RNN operator
 * \author Zixuan Wei
 */

#if MXNET_USE_ONEDNN == 1

#include <numeric>
#include <functional>

#include "dnnl_rnn-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLRnnParam);

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

void DNNLRnnLayerParam::SetDims() {
  const int ngates = GetRnnGatesNum(mode);
  //* NOTES: LBR-GRU's new gate formula needs two bias. So it has one more bias with LBR-GRU
  const int nbias         = mode == rnn_enum::kGru ? (ngates + 1) : ngates;
  const int num_direction = bidirectional ? 2 : 1;

  const int iter_size = proj_size < 0 ? state_size : proj_size;
  src_dims.assign({seq_len, batch_size, input_size});
  weight_layer_dims.assign({num_layer, num_direction, input_size, ngates, state_size});
  weight_iter_dims.assign({num_layer, num_direction, iter_size, ngates, state_size});
  weight_proj_dims.assign({num_layer, num_direction, state_size, iter_size});
  bias_dims.assign({num_layer, num_direction, nbias, state_size});
  dst_dims.assign({seq_len, batch_size, iter_size * num_direction});
  state_dims.assign({num_layer, num_direction, batch_size, iter_size});
  cell_dims.assign({num_layer, num_direction, batch_size, state_size});

  // unidirectional size of a single cell
  single_w_size = (input_size + iter_size) * ngates * state_size;
  if (proj_size > 0)
    single_w_size += state_size * proj_size;
  single_b_size        = nbias * state_size;
  native_single_b_size = ngates * state_size * 2;  // native RNN variants have double bias
  single_state_size    = batch_size * iter_size;

  // Get workspace size for cached weights memory
  // multiplication of tensor dimensions
  static auto tz_volume = [](const memory::dims& tz_dims) {
    return std::accumulate(tz_dims.begin(),
                           tz_dims.end(),
                           static_cast<memory::dim>(1),
                           std::multiplies<memory::dim>());
  };

  workspace_size =
      tz_volume(weight_layer_dims) + tz_volume(weight_iter_dims) + tz_volume(bias_dims);
  if (proj_size > 0)
    workspace_size += tz_volume(weight_proj_dims);
  reserve_size = 0;
}

DNNLRnnFullParam DNNLRnnFullParamParser(const NodeAttrs& attrs,
                                        const index_t seq_len,
                                        const index_t batch_size,
                                        const index_t input_size) {
  const RNNParam& rnn_param = nnvm::get<RNNParam>(attrs.parsed);
  DNNLRnnFullParam full_param;
  full_param.default_param = rnn_param;
  try {
    full_param.dnnl_param.Init(attrs.dict, dmlc::parameter::kAllowUnknown);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs.op->name << "("
       << "name=\"" << attrs.name << "\"";
    for (const auto& k : attrs.dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }

  const int state_size = rnn_param.state_size;
  const int proj_size =
      rnn_param.projection_size.has_value() ? rnn_param.projection_size.value() : -1;
  const int iter_size =
      rnn_param.projection_size.has_value() ? rnn_param.projection_size.value() : state_size;
  LayerParamVector& layer_params = full_param.layer_params;

  full_param.default_param.seq_length_ = seq_len;
  full_param.default_param.batch_size_ = batch_size;
  full_param.default_param.input_size_ = input_size;
  // Set basic size by constructing DNNLRnnLayerParam instance(s)
  if (rnn_param.bidirectional) {  // unfused bidirectional multi-layer RNN
    layer_params.emplace_back(
        1, batch_size, seq_len, input_size, state_size, proj_size, rnn_param.mode);
    for (size_t layer = 1; layer < rnn_param.num_layers; ++layer) {
      layer_params.emplace_back(
          1, batch_size, seq_len, iter_size * 2, state_size, proj_size, rnn_param.mode);
    }
  } else if (input_size == iter_size) {  // fused multi-layer
    layer_params.emplace_back(rnn_param.num_layers,
                              batch_size,
                              seq_len,
                              input_size,
                              state_size,
                              proj_size,
                              rnn_param.mode,
                              false);
  } else {  // unfused 1st layer, plus fused 2~end layers
    layer_params.emplace_back(
        1, batch_size, seq_len, input_size, state_size, proj_size, rnn_param.mode, false);
    if (rnn_param.num_layers > 1)
      layer_params.emplace_back(rnn_param.num_layers - 1,
                                batch_size,
                                seq_len,
                                iter_size,
                                state_size,
                                proj_size,
                                rnn_param.mode,
                                false);
  }

  // Set dims, workspace size, state_outputs, quantized and enable_u8_output flag
  for (auto& layer_param : layer_params) {
    layer_param.SetDims();
    layer_param.state_outputs    = rnn_param.state_outputs;
    layer_param.quantized        = full_param.dnnl_param.quantized;
    layer_param.enable_u8_output = true;
  }
  // Quantized RNN operator produces kFloat32 outputs.
  if (full_param.dnnl_param.quantized)
    layer_params.back().enable_u8_output = false;
  return full_param;
}

void DNNLRnnMemMgr::Init(const dim_t size, const Context& ctx) {
  workspace_ = NDArray(TShape({size}), ctx, false, mshadow::kUint8);
  if (workspace_.data().dptr_ == nullptr)
    LOG(FATAL) << "oneDNN RNN operator memory allocation error.";
  curr_mem  = static_cast<char*>(workspace_.data().dptr_);
  mem_size  = size;
  curr_size = size;
}

dnnl::memory* DNNLRnnMemMgr::Alloc(const dnnl::memory::desc& md) {
  if (curr_mem == nullptr) {
    curr_mem = static_cast<char*>(workspace_.data().dptr_);
  }

  dnnl_mem_ptr ret(new dnnl::memory());
  size_t addr       = reinterpret_cast<size_t>(curr_mem);
  size_t last_chunk = addr % alignment;
  size_t padding    = alignment - last_chunk;
  addr += padding;
  CHECK_EQ(addr % alignment, 0);

  curr_size -= (md.get_size() + padding);
  if (curr_size < 0) {
    ret.reset(new dnnl::memory(md, cpu_engine));
  } else {
    curr_mem += (md.get_size() + padding);
    ret.reset(new dnnl::memory(md, cpu_engine, reinterpret_cast<void*>(addr)));
  }
  RegisterMem(ret);
  return ret.get();
}

RnnPrimitive GetRnnFwdPrim(const DNNLRnnLayerParam& layer_param,
                           const bool is_train,
                           const NDArray& data,
                           const NDArray& params,
                           const shared_dnnl_attr_t attr) {
  using namespace dnnl;
  using tag                         = dnnl::memory::format_tag;
  const int mode                    = layer_param.mode;
  memory::data_type src_layer_dtype = get_dnnl_type(data.dtype());
  memory::data_type iter_dtype      = get_dnnl_type(mshadow::kFloat32);
  memory::data_type weight_dtype =
      get_dnnl_type(layer_param.quantized ? mshadow::kInt8 : params.dtype());
  memory::data_type bias_dtype = get_dnnl_type(mshadow::kFloat32);
  memory::data_type dst_layer_dtype =
      get_dnnl_type((layer_param.quantized && layer_param.enable_u8_output) ? mshadow::kUint8 :
                                                                              mshadow::kFloat32);

  const prop_kind prop = is_train ? prop_kind::forward_training : prop_kind::forward_inference;
  const rnn_direction dnnl_rnn_direction = layer_param.bidirectional ?
                                               rnn_direction::bidirectional_concat :
                                               rnn_direction::unidirectional;

  auto src_layer_desc    = memory::desc(layer_param.src_dims, src_layer_dtype, tag::tnc);
  auto weight_layer_desc = memory::desc(layer_param.weight_layer_dims, weight_dtype, tag::any);
  auto weight_iter_desc  = memory::desc(layer_param.weight_iter_dims, weight_dtype, tag::any);
  auto bias_desc         = memory::desc(layer_param.bias_dims, bias_dtype, tag::ldgo);
  auto dst_layer_desc    = memory::desc(layer_param.dst_dims, dst_layer_dtype, tag::tnc);
  auto src_state_desc    = memory::desc(layer_param.state_dims, iter_dtype, tag::ldnc);
  auto src_cell_desc     = memory::desc(layer_param.cell_dims, iter_dtype, tag::ldnc);
  auto weight_peep_desc  = memory::desc();
  auto weight_proj_desc  = layer_param.proj_size > 0 ?
                              memory::desc(layer_param.weight_proj_dims, weight_dtype, tag::any) :
                              memory::desc();
  auto dst_state_desc = layer_param.state_outputs ?
                            memory::desc(layer_param.state_dims, iter_dtype, tag::ldnc) :
                            memory::desc();
  auto dst_cell_desc = layer_param.state_outputs ?
                           memory::desc(layer_param.cell_dims, iter_dtype, tag::ldnc) :
                           memory::desc();

  auto fwd = RnnPrimitive();
  switch (mode) {
    case rnn_enum::kLstm:
      fwd = RnnPrimitive::Create<lstm_forward>(attr,
                                               prop,
                                               dnnl_rnn_direction,
                                               src_layer_desc,
                                               src_state_desc,
                                               src_cell_desc,
                                               weight_layer_desc,
                                               weight_iter_desc,
                                               weight_peep_desc,
                                               weight_proj_desc,
                                               bias_desc,
                                               dst_layer_desc,
                                               dst_state_desc,
                                               dst_cell_desc);
      break;
    case rnn_enum::kGru:
      fwd = RnnPrimitive::Create<lbr_gru_forward>(attr,
                                                  prop,
                                                  dnnl_rnn_direction,
                                                  src_layer_desc,
                                                  src_state_desc,
                                                  weight_layer_desc,
                                                  weight_iter_desc,
                                                  bias_desc,
                                                  dst_layer_desc,
                                                  dst_state_desc);
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      fwd = RnnPrimitive::Create<vanilla_rnn_forward>(
          attr,
          prop,
          mode == rnn_enum::kRnnTanh ? algorithm::eltwise_tanh : algorithm::eltwise_relu,
          dnnl_rnn_direction,
          src_layer_desc,
          src_state_desc,
          weight_layer_desc,
          weight_iter_desc,
          bias_desc,
          dst_layer_desc,
          dst_state_desc);
      break;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      break;
  }
  return fwd;
}

RnnBwdPrimitive GetRnnBwdPrim(const DNNLRnnForwardTraining& fwd,
                              const NDArray& data,
                              const NDArray& params) {
  using namespace dnnl;
  using tag                            = dnnl::memory::format_tag;
  const DNNLRnnLayerParam& layer_param = fwd.GetParam();
  const int mode                       = layer_param.mode;
  memory::data_type data_type          = get_dnnl_type(data.dtype());
  memory::data_type weight_type        = get_dnnl_type(params.dtype());
  const prop_kind prop                 = prop_kind::backward;
  rnn_direction dnnl_rnn_direction     = layer_param.bidirectional ?
                                         rnn_direction::bidirectional_concat :
                                         rnn_direction::unidirectional;

  auto src_layer_desc    = memory::desc(layer_param.src_dims, data_type, tag::tnc);
  auto weight_layer_desc = memory::desc(layer_param.weight_layer_dims, weight_type, tag::any);
  auto weight_iter_desc  = memory::desc(layer_param.weight_iter_dims, weight_type, tag::any);
  auto bias_desc         = memory::desc(layer_param.bias_dims, data_type, tag::ldgo);
  auto dst_layer_desc    = memory::desc(layer_param.dst_dims, data_type, tag::tnc);
  auto src_state_desc    = memory::desc(layer_param.state_dims, data_type, tag::ldnc);
  auto dst_state_desc    = layer_param.state_outputs ?
                            memory::desc(layer_param.state_dims, data_type, tag::ldnc) :
                            memory::desc();

  const void* fwd_pd = fwd.GetPrimDesc();
  auto bwd           = RnnBwdPrimitive();
  switch (mode) {
    case rnn_enum::kLstm: {
      const lstm_forward::primitive_desc* pd =
          reinterpret_cast<const lstm_forward::primitive_desc*>(fwd_pd);
      bwd = RnnBwdPrimitive::Create<lstm_forward, lstm_backward>(*pd,
                                                                 prop,
                                                                 dnnl_rnn_direction,
                                                                 // data desc
                                                                 src_layer_desc,
                                                                 src_state_desc,
                                                                 src_state_desc,
                                                                 weight_layer_desc,
                                                                 weight_iter_desc,
                                                                 bias_desc,
                                                                 dst_layer_desc,
                                                                 dst_state_desc,
                                                                 dst_state_desc,
                                                                 // diff desc
                                                                 src_layer_desc,
                                                                 src_state_desc,
                                                                 src_state_desc,
                                                                 weight_layer_desc,
                                                                 weight_iter_desc,
                                                                 bias_desc,
                                                                 dst_layer_desc,
                                                                 dst_state_desc,
                                                                 dst_state_desc);
    } break;
    case rnn_enum::kGru: {
      const lbr_gru_forward::primitive_desc* pd =
          reinterpret_cast<const lbr_gru_forward::primitive_desc*>(fwd_pd);
      bwd = RnnBwdPrimitive::Create<lbr_gru_forward, lbr_gru_backward>(*pd,
                                                                       prop,
                                                                       dnnl_rnn_direction,
                                                                       // data desc
                                                                       src_layer_desc,
                                                                       src_state_desc,
                                                                       weight_layer_desc,
                                                                       weight_iter_desc,
                                                                       bias_desc,
                                                                       dst_layer_desc,
                                                                       dst_state_desc,
                                                                       // diff desc
                                                                       src_layer_desc,
                                                                       src_state_desc,
                                                                       weight_layer_desc,
                                                                       weight_iter_desc,
                                                                       bias_desc,
                                                                       dst_layer_desc,
                                                                       dst_state_desc);
    } break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh: {
      const vanilla_rnn_forward::primitive_desc* pd =
          reinterpret_cast<const vanilla_rnn_forward::primitive_desc*>(fwd_pd);
      bwd = RnnBwdPrimitive::Create<vanilla_rnn_forward, vanilla_rnn_backward>(
          *pd,
          prop,
          mode == rnn_enum::kRnnTanh ? algorithm::eltwise_tanh : algorithm::eltwise_relu,
          dnnl_rnn_direction,
          // data desc
          src_layer_desc,
          src_state_desc,
          weight_layer_desc,
          weight_iter_desc,
          bias_desc,
          dst_layer_desc,
          dst_state_desc,
          // diff desc
          src_layer_desc,
          src_state_desc,
          weight_layer_desc,
          weight_iter_desc,
          bias_desc,
          dst_layer_desc,
          dst_state_desc);
    } break;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      break;
  }
  return bwd;
}

/*
 * Native weights layout is:
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
static void ConcatWeights(const dnnl::memory& dst,
                          const int concat_dimension,
                          const std::vector<void*>& src_ptrs,
                          const dnnl::memory::format_tag src_format) {
  using memory    = dnnl::memory;
  auto cpu_engine = dst.get_engine();
  dnnl::stream s(cpu_engine);
  const memory::desc& dst_desc = dst.get_desc();
  // Use dst memory dims to initialize src memory dims, then set the concat
  // dim to 1. And Rnn weights are 5-dimension tensor.
  memory::dims src_dims(dst_desc.data.dims, dst_desc.data.dims + dst_desc.data.ndims);
  src_dims.at(concat_dimension) = 1;
  std::vector<memory::desc> src_descs;
  std::unordered_map<int, memory> concat_args;

  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    src_descs.emplace_back(
        src_dims, static_cast<memory::data_type>(dst_desc.data.data_type), src_format);
    concat_args.emplace(DNNL_ARG_MULTIPLE_SRC + i,
                        memory(src_descs.back(), cpu_engine, src_ptrs.at(i)));
  }
  concat_args.emplace(DNNL_ARG_DST, dst);

  auto concat_pd =
      dnnl::concat::primitive_desc(dst.get_desc(), concat_dimension, src_descs, cpu_engine);
  dnnl::concat(concat_pd).execute(s, concat_args);
}

#define RNN_HANDLE_FUNC_NAME set_handle
#define RNN_HANDLE_FUNC(RNN_FUNC_NAME)                                                    \
  auto RNN_FUNC_NAME = [&cpu_engine, &args](int arg_name, const desc& md, void* handle) { \
    if (args.find(arg_name) != args.end()) {                                              \
      if (handle != nullptr)                                                              \
        args.at(arg_name).set_data_handle(handle);                                        \
    } else {                                                                              \
      args[arg_name] =                                                                    \
          handle ? dnnl::memory(md, cpu_engine, handle) : dnnl::memory(md, cpu_engine);   \
    }                                                                                     \
  }

#define RNN_FWD_SET(NAME, DIMS, TAG, HANDLE, DTYPE) \
  RNN_FWD_SET_(RNN_HANDLE_FUNC_NAME, NAME, DIMS, TAG, HANDLE, DTYPE)

#define RNN_FWD_SET_(FUNC, NAME, DIMS, TAG, HANDLE, DTYPE) \
  FUNC(DNNL_ARG_##NAME, {DIMS, get_dnnl_type(DTYPE), TAG}, HANDLE)

#define RNN_BWD_SET(NAME, ARGS, HANDLE) RNN_BWD_SET_(RNN_HANDLE_FUNC_NAME, NAME, ARGS, HANDLE)

#define RNN_BWD_SET_(FUNC, NAME, ARGS, HANDLE) \
  FUNC(DNNL_ARG_DIFF_##NAME, ARGS.at(DNNL_ARG_##NAME).get_desc(), HANDLE)

/*
 * Set new src data handler to Forward memory. The memory primitives are
 * not initialized until SetNewDataMem is first invoked. Src data handler
 * must not be nullptr, except for cx with LSTM mode. If either hy, cy is
 * nullptr, it may run with non-state_ouput or non-LSTM mode. Thus, the
 * corresponding memory should be a empty dnnl::memory().
 */
void DNNLRnnForward::SetNewDataMem(void* x,
                                   void* hx,
                                   void* cx,
                                   void* y,
                                   void* hy,
                                   void* cy,
                                   const int dtype) {
  using desc            = dnnl::memory::desc;
  using format_tag      = dnnl::memory::format_tag;
  auto& cpu_engine      = CpuEngine::Get()->get_engine();
  dnnl_args_map_t& args = net_args_;

  int src_dtype = dtype;
  int dst_dtype = dtype;
  if (param_.quantized) {
    src_dtype = mshadow::kUint8;
    if (param_.enable_u8_output)
      dst_dtype = mshadow::kUint8;
  }

  RNN_HANDLE_FUNC(RNN_HANDLE_FUNC_NAME);

  // Set various data memory
  RNN_FWD_SET(SRC, param_.src_dims, format_tag::tnc, x, src_dtype);
  RNN_FWD_SET(DST, param_.dst_dims, format_tag::tnc, y, dst_dtype);
  RNN_FWD_SET(SRC_ITER, param_.state_dims, format_tag::ldnc, hx, dtype);

  if (param_.state_outputs) {
    RNN_FWD_SET(DST_ITER, param_.state_dims, format_tag::ldnc, hy, dtype);
  }

  if (param_.mode == rnn_enum::kLstm) {
    RNN_FWD_SET(SRC_ITER_C, param_.cell_dims, format_tag::ldnc, cx, dtype);
    if (param_.state_outputs) {
      RNN_FWD_SET(DST_ITER_C, param_.cell_dims, format_tag::ldnc, cy, dtype);
    }
  }
}

inline void DNNLMemoryReorder(const dnnl::memory& src, const dnnl::memory& dst) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, dnnl::reorder, OpHash> reorderPrimitives;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, dnnl::reorder, OpHash> reorderPrimitives;
#endif
  OpSignature key{};
  key.AddSign(src);
  key.AddSign(dst);

  auto it = reorderPrimitives.find(key);
  if (it == reorderPrimitives.end()) {
    auto reorder = dnnl::reorder(src, dst);
    it           = AddToCache(&reorderPrimitives, key, reorder);
  }

  dnnl_args_map_t net_args;
  net_args.emplace(DNNL_ARG_SRC, src);
  net_args.emplace(DNNL_ARG_DST, dst);
  DNNLStream::Get()->RegisterPrimArgs(it->second, net_args);
}

/*
 * Reorder the concatenated weights memory to a efficient memory block
 * with primitive-prefered format.
 */
void DNNLRnnForward::ReorderWeights() {
  if (param_.quantized) {
    const dnnl::primitive_attr& attr = this->fwd_inf_.GetPrimAttr();
    auto ReorderWithAttr             = [&](dnnl::memory& src, dnnl::memory& dst) {
      auto reorder_pd = dnnl::reorder::primitive_desc(src, dst, attr);
      dnnl_args_map_t net_args;
      net_args[DNNL_ARG_SRC] = src;
      net_args[DNNL_ARG_DST] = dst;
      DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd), net_args);
    };
    ReorderWithAttr(*weights_layer_r_, *weights_layer_);
    ReorderWithAttr(*weights_iter_r_, *weights_iter_);
    if (param_.proj_size > 0)
      ReorderWithAttr(*weights_proj_r_, *weights_proj_);
  } else {
    DNNLMemoryReorder(*weights_layer_r_, *weights_layer_);
    DNNLMemoryReorder(*weights_iter_r_, *weights_iter_);
    if (param_.proj_size > 0)
      DNNLMemoryReorder(*weights_proj_r_, *weights_proj_);
  }
}

void AdjustGruGateOrder(char* weight,
                        const size_t input_size,
                        const size_t hidden_size,
                        const int dtype) {
  // mxnet gru gate order is reset, update and new gates
  // dnnl gru gate order is update, reset and new gates
  size_t single_weight_bytes = input_size * hidden_size * mshadow::mshadow_sizeof(dtype);
  char* weight_reset         = weight;
  char* weight_update        = weight + single_weight_bytes;
  std::swap_ranges(weight_reset, weight_update, weight_update);
}

/*
 * Fuse uni-directional bias among single layer.
 */
template <typename DType>
void FuseBias(DType* fuse_bias, DType* native_bias, const int mode, const size_t state_size) {
  const size_t ngates   = GetRnnGatesNum(mode);
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const size_t nbias    = mode == rnn_enum::kGru ? ngates + 1 : ngates;
  // MSVC-14.0 (OpenMP 2.0 compatible) doesn't support unsigned integral type in
  // OpenMP 'for' statement.
  const int state_size_ = static_cast<int>(state_size);
  const int single_b_sz = static_cast<int>(nbias * state_size);
  DType* bx             = native_bias;
  DType* bh             = native_bias + state_size * ngates;
  if (mode == rnn_enum::kGru) {
// While mxnet gru gate order is reset, update and new gates,
// dnnl gru gate order is update, reset and new gates. So
// we need to swap the order of reset and update from mxnet.
#pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < state_size_; j++) {
      // Swap summed reset, update bias
      fuse_bias[j + state_size] = bx[j] + bh[j];
      fuse_bias[j]              = bx[j + state_size] + bh[j + state_size];

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

inline void EmplaceNetArgs(dnnl_args_map_t* net_args, const int arg_name, const dnnl::memory* mem) {
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
 * Copy native memory to dnnl-format memory. It will initialize the memory
 * when first invoked. Then, the native weight_layer and weight_iter are
 * concatenated to xxx_xx_r memory. Per the different gates order of GRU,
 * it will swap the memory blocks of gates among concatenated memory
 * inplace. From then on, the xxx_xx_r memory is reordered to target
 * memory with preferred format_tag. Finally, native bias is fused to DNNL
 * bias memory.
 */
void DNNLRnnForward::SetWeightsMem(void* w_ptr, void* b_ptr, const bool is_train, const int dtype) {
  using format_tag         = dnnl::memory::format_tag;
  const auto dnnl_dtype    = get_dnnl_type(dtype);
  const size_t dtype_bytes = mshadow::mshadow_sizeof(dtype);

  const size_t buffer_bytes =
      this->GetSize()  // byte number of the buffer
      + (param_.workspace_size + param_.reserve_size) * dtype_bytes +
      kDNNLAlign * 7;  // Add margin for alignment of seven times allocation for the
                       // dnnl memory handlers, i.e. weights_layer_, weights_iter_,
                       // weights_proj_, bias_, weights_layer_r_, weights_iter_r_,
                       // and weights_proj_r_.
  if (mem_mgr_.Size() < buffer_bytes)
    mem_mgr_.Init(buffer_bytes, this->ctx_);

  const bool use_proj = (param_.proj_size > 0);
  // Get the weights' memory for RNN forward primitive
  if (weights_layer_ == nullptr) {
    weights_layer_ = mem_mgr_.Alloc(fwd_inf_.GetLayerDesc());
  }
  if (weights_iter_ == nullptr) {
    weights_iter_ = mem_mgr_.Alloc(fwd_inf_.GetIterDesc());
  }
  if (use_proj && weights_proj_ == nullptr) {
    weights_proj_ = mem_mgr_.Alloc(fwd_inf_.GetProjDesc());
  }
  if (bias_ == nullptr) {
    bias_ = mem_mgr_.Alloc({param_.bias_dims, dnnl_dtype, format_tag::ldgo});
  }

  // Get the intermediate memory for weights concat & reorder
  if (weights_layer_r_ == nullptr) {
    weights_layer_r_ = mem_mgr_.Alloc({param_.weight_layer_dims, dnnl_dtype, format_tag::ldgoi});
  }
  if (weights_iter_r_ == nullptr) {
    weights_iter_r_ = mem_mgr_.Alloc({param_.weight_iter_dims, dnnl_dtype, format_tag::ldgoi});
  }
  if (use_proj && weights_proj_r_ == nullptr) {
    weights_proj_r_ = mem_mgr_.Alloc({param_.weight_proj_dims, dnnl_dtype, format_tag::ldoi});
  }

  // convert void* to char* for arithmetic operations
  const size_t iter_size = use_proj ? param_.proj_size : param_.state_size;
  char* weights_ptr      = static_cast<char*>(w_ptr);
  size_t wx_bytes        = GetRnnGatesNum(param_.mode) * param_.state_size * param_.input_size *
                    dtype_bytes;  //* DIMS: ngates x state_size x input_size
  size_t wh_bytes = GetRnnGatesNum(param_.mode) * param_.state_size * iter_size *
                    dtype_bytes;  //* DIMS: ngates x state_size x state_size, if not use projection.
                                  // With projection, DIMS is ngates x state_size x projection_size
  size_t wr_bytes = param_.state_size * iter_size * dtype_bytes;
  char* l2r_wx    = weights_ptr;
  char* l2r_wh    = l2r_wx + wx_bytes;  //* DIMS: ngates x state_size * state_size
  char* l2r_wr    = l2r_wh + wh_bytes;  //* DIMS: ngates x state_size * iter_size

  if (param_.num_layer == 1 && param_.bidirectional) {
    //* single bidirectinal layer, concat weights on direction axis
    char* r2l_wx = weights_ptr + param_.single_w_size * dtype_bytes;
    char* r2l_wh = r2l_wx + wx_bytes;  //* DIMS: ngates x state_size x state_size
    char* r2l_wr = r2l_wh + wh_bytes;  //* DIMS: ngates x state_size x iter_size
    ConcatWeights(*weights_layer_r_, 1, {l2r_wx, r2l_wx}, format_tag::ldgoi);
    ConcatWeights(*weights_iter_r_, 1, {l2r_wh, r2l_wh}, format_tag::ldgoi);
    if (use_proj)
      ConcatWeights(*weights_proj_r_, 1, {l2r_wr, r2l_wr}, format_tag::ldoi);
  } else if (param_.num_layer == 1 && !param_.bidirectional) {
    //* single uni-directional layer, no concatenate operator needed
    std::memcpy(weights_layer_r_->get_data_handle(), l2r_wx, wx_bytes);
    std::memcpy(weights_iter_r_->get_data_handle(), l2r_wh, wh_bytes);
    if (use_proj)
      std::memcpy(weights_proj_r_->get_data_handle(), l2r_wr, wr_bytes);
  } else if (param_.num_layer > 1 && !param_.bidirectional) {
    //* concat fused multi-layer weights on layer axis
    std::vector<void*> l2r_wx_ptrs;
    std::vector<void*> l2r_wh_ptrs;
    std::vector<void*> l2r_wr_ptrs;
    for (int lyr = 0; lyr < param_.num_layer; ++lyr) {
      char* lth_wx = l2r_wx + lyr * param_.single_w_size * dtype_bytes;
      char* lth_wh = lth_wx + wx_bytes;
      char* lth_wr = lth_wh + wh_bytes;
      l2r_wx_ptrs.push_back(lth_wx);
      l2r_wh_ptrs.push_back(lth_wh);
      if (use_proj)
        l2r_wr_ptrs.push_back(lth_wr);
    }
    ConcatWeights(*weights_layer_r_, 0, l2r_wx_ptrs, format_tag::ldgoi);
    ConcatWeights(*weights_iter_r_, 0, l2r_wh_ptrs, format_tag::ldgoi);
    if (use_proj)
      ConcatWeights(*weights_proj_r_, 0, l2r_wr_ptrs, format_tag::ldoi);
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

  // Process bias
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    DType* native_b_ptr = static_cast<DType*>(b_ptr);
    DType* fused_bias   = static_cast<DType*>(bias_->get_data_handle());
    for (int lyr = 0; lyr < param_.num_layer; ++lyr) {
      for (int d = 0; d < param_.bidirectional + 1; ++d) {
        FuseBias<DType>(fused_bias, native_b_ptr, param_.mode, param_.state_size);
        fused_bias += param_.single_b_size;
        native_b_ptr += param_.native_single_b_size;
      }
    }
  });

  // insert weights into net_args
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_WEIGHTS_LAYER, this->weights_layer_);
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_WEIGHTS_ITER, this->weights_iter_);
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_BIAS, this->bias_);
  if (use_proj)
    EmplaceNetArgs(&this->net_args_, DNNL_ARG_WEIGHTS_PROJECTION, this->weights_proj_);

  if (!is_train) {
    // Reorder after adjustment only when is_train == false. When is_train == true, i.e.
    // in forward training path, we use plain memory (ldxxx) as the space for weights and
    // their gradients. Then, forward training primitives could fetch them from the scope
    // of forward inference. And from there, we don't need to reorder the plain memory to
    // the optimal rnn-packed memory for forward inference
    ReorderWeights();
    initialized_ = true;
  }
}

void DNNLRnnForwardTraining::SetTrnMem(const DNNLRnnForward& fwd) {
  using memory           = dnnl::memory;
  const auto& cpu_engine = CpuEngine::Get()->get_engine();
  auto s                 = dnnl::stream(cpu_engine);
  // Prepare dnnl::memorys for weights_layer, weight_iter, and workspace
  if (workspace_ == nullptr)
    workspace_ = dnnl_shared_mem_t(new memory(fwd_trn_.GetWorkspaceDesc(), cpu_engine));
  if (weights_layer_ == nullptr)
    weights_layer_ = dnnl_shared_mem_t(new memory(fwd_trn_.GetLayerDesc(), cpu_engine));
  if (weights_iter_ == nullptr)
    weights_iter_ = dnnl_shared_mem_t(new memory(fwd_trn_.GetIterDesc(), cpu_engine));

  // fill weights memory using the reordered weights of fwd_inference primitive
  if (fwd.weights_layer_r_->get_desc() == fwd_trn_.GetLayerDesc()) {
    weights_layer_->set_data_handle(fwd.weights_layer_r_->get_data_handle());
  } else {
    DNNLMemoryReorder(*fwd.weights_layer_r_, *weights_layer_);
  }

  if (fwd.weights_iter_r_->get_desc() == fwd_trn_.GetIterDesc()) {
    weights_iter_->set_data_handle(fwd.weights_iter_r_->get_data_handle());
  } else {
    DNNLMemoryReorder(*fwd.weights_iter_r_, *weights_iter_);
  }

  // bias are always in format_tag::ldgo
  this->bias_ = fwd.bias_;

  // insert weights into net_args
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_WEIGHTS_LAYER, this->weights_layer_.get());
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_WEIGHTS_ITER, this->weights_iter_.get());
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_BIAS, this->bias_);
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_WORKSPACE, this->workspace_.get());
}

void DNNLRnnForwardTraining::FetchData(const DNNLRnnForward& fwd) {
  for (auto& kv : fwd.net_args_) {
    switch (kv.first) {
      case DNNL_ARG_WEIGHTS_LAYER:
      case DNNL_ARG_WEIGHTS_ITER:
      case DNNL_ARG_BIAS:
      case DNNL_ARG_WORKSPACE:
        continue;

      default:
        EmplaceNetArgs(&this->net_args_, kv.first, &kv.second);
    }
  }
}

void DNNLRnnOp::Init(const OpContext& op_ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  using format_tag = dnnl::memory::format_tag;

  // Get the bytes of a real type
  const NDArray& weights        = inputs[rnn_enum::kParams];
  int dtype                     = weights.dtype();
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
  const bool is_training  = (op_ctx.is_train || op_ctx.need_grad);
  const size_t num_fusion = full_param_.layer_params.size();
  const Context& ctx      = op_ctx.run_ctx.ctx;
  if (fwd_inf_vec_.size() < num_fusion) {
    for (auto& layer_param : full_param_.layer_params) {
      fwd_inf_vec_.emplace_back(
          ctx, layer_param, false, inputs[rnn_enum::kData], inputs[rnn_enum::kParams], nullptr);
    }
  }

  if (is_training && fwd_trn_vec_.size() < num_fusion) {
    for (auto& layer_param : full_param_.layer_params) {
      fwd_trn_vec_.emplace_back(
          layer_param, true, inputs[rnn_enum::kData], inputs[rnn_enum::kParams]);
    }
  }

  for (auto& fwd_layer : fwd_inf_vec_) {
    size_t single_w_bytes      = fwd_layer.GetParam().single_w_size * dtype_bytes;
    size_t single_b_bytes      = fwd_layer.GetParam().native_single_b_size * dtype_bytes;
    size_t directions          = fwd_layer.GetParam().bidirectional ? 2 : 1;
    size_t layer_weights_bytes = single_w_bytes * directions;
    size_t layer_bias_bytes    = single_b_bytes * directions;  // Native MXNet has double bias

    if (!fwd_layer.IsInitialized() || is_training)
      fwd_layer.SetWeightsMem(weights_ptr, bias_ptr, is_training, dtype);
    weights_ptr += layer_weights_bytes;
    bias_ptr += layer_bias_bytes;
  }

  if (is_training) {
    CHECK_EQ(fwd_trn_vec_.size(), fwd_inf_vec_.size())
        << "Layers' configurations of forward inference and forward training are disparate.";
    for (size_t lyr = 0; lyr < fwd_inf_vec_.size(); ++lyr)
      fwd_trn_vec_.at(lyr).SetTrnMem(fwd_inf_vec_.at(lyr));
  }

  CHECK_EQ(num_fusion, fwd_inf_vec_.size())
      << "Layer vector's size has a different value than the number of fusion.";
  if (dst_.size() < num_fusion - 1) {
    const int data_dtype     = outputs[rnn_enum::kOut].dtype();
    const size_t data_dbytes = mshadow::mshadow_sizeof(data_dtype);
    mgr_.Init((outputs[rnn_enum::kOut].data().Size() * data_dbytes + kDNNLAlign) * (num_fusion - 1),
              op_ctx.run_ctx.ctx);
    // Here we need `fwd_inf_vec_.size() - 1` spaces for the intermediate results of the multiple
    // fused layers. And for the result of the last fused layer, `outputs[rnn_enum::kOut]` could
    // provide the space. Hence, `forward_inf_vec_.back()` is excluded when allocates the spaces
    // for intermediate results.
    for (std::vector<DNNLRnnForward>::const_iterator fwd = fwd_inf_vec_.begin();
         fwd != fwd_inf_vec_.end() - 1;
         ++fwd)
      dst_.push_back(
          mgr_.Alloc({fwd->GetParam().dst_dims, get_dnnl_type(data_dtype), format_tag::tnc}));
  }

  if (!is_training)
    initialized_ = true;
}

void DNNLRnnBackward::FetchDataWeightsMem(const DNNLRnnForwardTraining& fwd) {
  using memory     = dnnl::memory;
  auto& cpu_engine = CpuEngine::Get()->get_engine();

  if (this->weights_layer_ == nullptr || this->weights_iter_ == nullptr) {
    this->weights_layer_ = dnnl_shared_mem_t(new memory(bwd_.weights_layer_desc_, cpu_engine));
    this->weights_iter_  = dnnl_shared_mem_t(new memory(bwd_.weights_iter_desc_, cpu_engine));
  }

  for (auto& kv : fwd.net_args_) {
    const dnnl::memory* valid_mem;
    switch (kv.first) {
      case DNNL_ARG_WEIGHTS_LAYER: {
        if (bwd_.weights_layer_desc_ == fwd.fwd_trn_.GetLayerDesc()) {
          this->weights_layer_->set_data_handle(kv.second.get_data_handle());
        } else {
          DNNLMemoryReorder(*fwd.weights_layer_, *this->weights_layer_);
        }
        valid_mem = this->weights_layer_.get();
      } break;
      case DNNL_ARG_WEIGHTS_ITER: {
        if (bwd_.weights_iter_desc_ == fwd.fwd_trn_.GetIterDesc()) {
          this->weights_iter_->set_data_handle(kv.second.get_data_handle());
        } else {
          DNNLMemoryReorder(*fwd.weights_iter_, *this->weights_iter_);
        }
        valid_mem = this->weights_iter_.get();
      } break;

      default:
        valid_mem = &kv.second;
    }
    EmplaceNetArgs(&this->net_args_, kv.first, valid_mem);
  }
}

void DNNLRnnBackward::SetWeightsGradsMem() {
  using tag = dnnl::memory::format_tag;

  if (this->diff_weights_layer_ == nullptr || this->diff_weights_iter_ == nullptr ||
      this->diff_bias_ == nullptr) {
    const auto& cpu_engine         = CpuEngine::Get()->get_engine();
    const DNNLRnnLayerParam& param = fwd_ptr_->GetParam();
    const auto dnnl_type =
        static_cast<dnnl::memory::data_type>(bwd_.diff_weights_layer_desc_.data.data_type);

    auto native_layer_desc = dnnl::memory::desc(param.weight_layer_dims, dnnl_type, tag::ldgoi);
    auto native_iter_desc  = dnnl::memory::desc(param.weight_iter_dims, dnnl_type, tag::ldgoi);

    this->diff_weights_layer_r_ = std::make_shared<dnnl::memory>(native_layer_desc, cpu_engine);
    this->diff_weights_iter_r_  = std::make_shared<dnnl::memory>(native_iter_desc, cpu_engine);

    if (native_layer_desc == bwd_.diff_weights_layer_desc_) {
      this->diff_weights_layer_ = std::make_shared<dnnl::memory>(
          bwd_.diff_weights_layer_desc_, cpu_engine, diff_weights_layer_r_->get_data_handle());
    } else {
      this->diff_weights_layer_ =
          std::make_shared<dnnl::memory>(bwd_.diff_weights_layer_desc_, cpu_engine);
    }
    if (native_iter_desc == bwd_.diff_weights_iter_desc_) {
      this->diff_weights_iter_ = std::make_shared<dnnl::memory>(
          bwd_.diff_weights_iter_desc_, cpu_engine, diff_weights_iter_r_->get_data_handle());
    } else {
      this->diff_weights_iter_ =
          std::make_shared<dnnl::memory>(bwd_.diff_weights_iter_desc_, cpu_engine);
    }
    this->diff_bias_ = std::make_shared<dnnl::memory>(bwd_.diff_bias_desc_, cpu_engine);
  }
  std::memset(
      this->diff_weights_layer_->get_data_handle(), 0, bwd_.diff_weights_layer_desc_.get_size());
  std::memset(
      this->diff_weights_iter_->get_data_handle(), 0, bwd_.diff_weights_iter_desc_.get_size());
  std::memset(this->diff_bias_->get_data_handle(), 0, bwd_.diff_bias_desc_.get_size());
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_DIFF_WEIGHTS_LAYER, this->diff_weights_layer_.get());
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_DIFF_WEIGHTS_ITER, this->diff_weights_iter_.get());
  EmplaceNetArgs(&this->net_args_, DNNL_ARG_DIFF_BIAS, this->diff_bias_.get());
}

void DNNLRnnBackward::SetDataGradsMem(void* diff_src,
                                      void* diff_state,
                                      void* diff_statecell,
                                      void* diff_dst,
                                      void* diff_state_out,
                                      void* diff_statecell_out,
                                      const int dtype) {
  using desc            = dnnl::memory::desc;
  auto& cpu_engine      = CpuEngine::Get()->get_engine();
  dnnl_args_map_t& args = this->net_args_;

  RNN_HANDLE_FUNC(RNN_HANDLE_FUNC_NAME);

  // Set various diff memory
  auto& fwd_args = fwd_ptr_->GetArgsMap();
  RNN_BWD_SET(SRC, fwd_args, diff_src);
  RNN_BWD_SET(SRC_ITER, fwd_args, diff_state);
  RNN_BWD_SET(DST, fwd_args, diff_dst);

  if (fwd_ptr_->GetParam().state_outputs)
    RNN_BWD_SET(DST_ITER, fwd_args, diff_state_out);

  if (fwd_ptr_->GetParam().mode == rnn_enum::kLstm) {
    RNN_BWD_SET(SRC_ITER_C, fwd_args, diff_statecell);
    if (fwd_ptr_->GetParam().state_outputs) {
      RNN_BWD_SET(DST_ITER_C, fwd_args, diff_statecell_out);
    }
  }
}

void DNNLRnnBackward::SetNativeWeightsGrads() const {
  if (this->diff_weights_layer_->get_desc() != this->diff_weights_layer_r_->get_desc()) {
    DNNLMemoryReorder(*this->diff_weights_layer_, *this->diff_weights_layer_r_);
  }
  if (this->diff_weights_iter_->get_desc() != this->diff_weights_iter_r_->get_desc()) {
    DNNLMemoryReorder(*this->diff_weights_iter_, *this->diff_weights_iter_r_);
  }
}

#define OPREQTYPE_SWITCH(ReqType, DType, FWrapper, ...)           \
  std::function<void(DType*, DType*, size_t)> FWrapper = nullptr; \
  if (kWriteTo == ReqType || kWriteInplace == ReqType)            \
    FWrapper = common::ParallelCopy<DType>;                       \
  else                                                            \
    FWrapper = common::ParallelAdd<DType>;                        \
  { __VA_ARGS__ }

void DNNLRnnBackward::CommitWeightsGrads(void* diff_weights,
                                         void* diff_bias,
                                         const OpReqType req,
                                         const int dtype) {
  const DNNLRnnLayerParam& param = fwd_ptr_->GetParam();

  void* diff_weights_layer_ptr = this->diff_weights_layer_->get_data_handle();
  void* diff_weights_iter_ptr  = this->diff_weights_iter_->get_data_handle();
  if (this->diff_weights_layer_->get_desc() != this->diff_weights_layer_r_->get_desc())
    diff_weights_layer_ptr = this->diff_weights_layer_r_->get_data_handle();
  if (this->diff_weights_iter_->get_desc() != this->diff_weights_iter_r_->get_desc())
    diff_weights_iter_ptr = this->diff_weights_iter_r_->get_data_handle();

  const int num_layer   = param.num_layer;
  const int direction   = param.bidirectional ? 2 : 1;
  const int ngates      = GetRnnGatesNum(param.mode);
  const size_t wxh_size = param.single_w_size;
  const size_t wx_size  = param.input_size * param.state_size * ngates;
  const size_t wh_size  = param.state_size * param.state_size * ngates;

  /* native weights layout is:
          1st-layer: | wx_lr  | wh_lr  | wx_rl | wh_rl |
          2st-layer: | wx_lr  | wh_lr  | wx_rl | wh_rl |
  size:              |    wxh_bytes    |
                     |wx_bytes|wh_bytes|
  */
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    DType* native_weights = static_cast<DType*>(diff_weights);
    DType* diff_wx_ptr    = static_cast<DType*>(diff_weights_layer_ptr);
    DType* diff_wh_ptr    = static_cast<DType*>(diff_weights_iter_ptr);
    OPREQTYPE_SWITCH(req, DType, FAccGrad, {
      if (param.mode != rnn_enum::kGru) {
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          FAccGrad(native_weights + shift * wxh_size, diff_wx_ptr + shift * wx_size, wx_size);
        }
        // align native_weights to weights_iter memory
        native_weights += wx_size;
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          FAccGrad(native_weights + shift * wxh_size, diff_wh_ptr + shift * wh_size, wh_size);
        }
      } else {
        const size_t wx_size_per_gate = param.input_size * param.state_size;
        const size_t wh_size_per_gate = param.state_size * param.state_size;
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          FAccGrad(native_weights + shift * wxh_size + wx_size_per_gate,
                   diff_wx_ptr + shift * wx_size,
                   wx_size_per_gate);
          FAccGrad(native_weights + shift * wxh_size,
                   diff_wx_ptr + shift * wx_size + wx_size_per_gate,
                   wx_size_per_gate);
          FAccGrad(native_weights + shift * wxh_size + 2 * wx_size_per_gate,
                   diff_wx_ptr + shift * wx_size + 2 * wx_size_per_gate,
                   wx_size_per_gate);
        }
        // align native_weights to weights_iter memory
        native_weights += wx_size;
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          FAccGrad(native_weights + shift * wxh_size + wh_size_per_gate,
                   diff_wh_ptr + shift * wh_size,
                   wh_size_per_gate);
          FAccGrad(native_weights + shift * wxh_size,
                   diff_wh_ptr + shift * wh_size + wh_size_per_gate,
                   wh_size_per_gate);
          FAccGrad(native_weights + shift * wxh_size + 2 * wh_size_per_gate,
                   diff_wh_ptr + shift * wh_size + 2 * wh_size_per_gate,
                   wh_size_per_gate);
        }
      }
    });
  });

  const size_t bias_size        = param.single_b_size;
  const size_t native_bias_size = param.native_single_b_size;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    DType* native_bias   = static_cast<DType*>(diff_bias);
    DType* diff_bias_ptr = static_cast<DType*>(this->diff_bias_->get_data_handle());
    OPREQTYPE_SWITCH(req, DType, FAccGrad, {
      if (param.mode != rnn_enum::kGru) {
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          FAccGrad(
              native_bias + shift * native_bias_size, diff_bias_ptr + shift * bias_size, bias_size);
          FAccGrad(native_bias + shift * native_bias_size + bias_size,
                   diff_bias_ptr + shift * bias_size,
                   bias_size);
        }
      } else {
        const size_t bias_size_per_gate = param.state_size;
        for (int shift = 0; shift < num_layer * direction; ++shift) {
          DType* native_reset  = native_bias + shift * native_bias_size;
          DType* native_update = native_reset + bias_size_per_gate;
          DType* update        = diff_bias_ptr + shift * bias_size;
          DType* reset         = update + bias_size_per_gate;

          FAccGrad(native_update, update, bias_size_per_gate);
          FAccGrad(native_reset, reset, bias_size_per_gate);
          FAccGrad(native_update + native_bias_size / 2, update, bias_size_per_gate);
          FAccGrad(native_reset + native_bias_size / 2, reset, bias_size_per_gate);

          DType* native_new_bx = native_update + bias_size_per_gate;
          DType* native_new_bh = native_new_bx + native_bias_size / 2;
          DType* new_bx        = reset + bias_size_per_gate;
          DType* new_bh        = new_bx + bias_size_per_gate;
          FAccGrad(native_new_bx, new_bx, bias_size_per_gate);
          FAccGrad(native_new_bh, new_bh, bias_size_per_gate);
        }
      }
    });
  });
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

void DNNLRnnOp::Forward(const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[1]);
  // In the `autograd.record()` context, RNNOp is required to run into
  // forward_training mode.
  const bool is_training        = (ctx.is_train || ctx.need_grad);
  const RNNParam& default_param = full_param_.default_param;
  if (is_training && default_param.projection_size.has_value())
    LOG(FATAL) << "Backward/Training mode is not implemented!";

  // Initialize weights version
  if (!initialized_ && weights_version_ == 0) {
    weights_version_ = inputs[rnn_enum::kParams].version();
  }

  // Check if weights NDArray was changed. If so, reset initialized_
  if (!is_training && fwd_inf_vec_.size() > 0 &&
      weights_version_ != inputs[rnn_enum::kParams].version()) {
    initialized_ = false;
    for (auto& fwd : fwd_inf_vec_)
      fwd.Reset();
    weights_version_ = inputs[rnn_enum::kParams].version();
  }

  if (dmlc::GetEnv("MXNET_RNN_USE_WEIGHT_CACHE", 0) && !initialized_) {
    LOG(INFO) << "The current weight of RNN is assumed to be fixed and cached during "
                 "the whole inference pipeline. Please set MXNET_RNN_USE_WEIGHT_CACHE=0, if "
                 "the weight changed at runtime.";
  }
  if ((!dmlc::GetEnv("MXNET_RNN_USE_WEIGHT_CACHE", 0) && !initialized_) || is_training ||
      fwd_inf_vec_.size() == 0) {
    Init(ctx, inputs, req, outputs);
  }

  // Get data type
  int data_dtype = outputs[rnn_enum::kOut].dtype();
  // Get temporary memory for output, state_out, statecell_out
  const int num_layers = default_param.num_layers;
  const int seq_length = default_param.seq_length_;
  const int batch_size = default_param.batch_size_;
  const int state_size = default_param.state_size;
  const int iter_size  = default_param.projection_size.has_value() ?
                            default_param.projection_size.value() :
                            default_param.state_size;
  const int directions = default_param.bidirectional ? 2 : 1;
  dnnl::memory::desc dst_desc({seq_length, batch_size, directions * iter_size},
                              get_dnnl_type(data_dtype),
                              dnnl::memory::format_tag::tnc);
  dnnl::memory::desc state_desc({num_layers, directions, batch_size, iter_size},
                                get_dnnl_type(data_dtype),
                                dnnl::memory::format_tag::ldnc);
  dnnl::memory::desc cell_desc({num_layers, directions, batch_size, state_size},
                               get_dnnl_type(data_dtype),
                               dnnl::memory::format_tag::ldnc);
  auto out_mem = CreateDNNLMem(outputs[rnn_enum::kOut], dst_desc, req[rnn_enum::kOut]);
  dnnl_output_t stateout_mem;
  dnnl_output_t statecellout_mem;

  // Get input & output NDArray
  char* src            = static_cast<char*>(inputs[rnn_enum::kData].data().dptr_);
  char* src_state      = static_cast<char*>(inputs[rnn_enum::kState].data().dptr_);
  char* dst            = static_cast<char*>(out_mem.second->get_data_handle());
  char* dst_state      = nullptr;  // Output state
  char* src_state_cell = nullptr;  // Used in LSTM for cell state
  char* dst_state_cell = nullptr;  // Used in LSTM for cell state

  if (default_param.state_outputs && req[rnn_enum::kStateOut] != kNullOp) {
    stateout_mem =
        CreateDNNLMem(outputs[rnn_enum::kStateOut], state_desc, req[rnn_enum::kStateOut]);
    dst_state = static_cast<char*>(stateout_mem.second->get_data_handle());
  }

  if (default_param.mode == rnn_enum::kLstm) {
    src_state_cell = static_cast<char*>(inputs[rnn_enum::kStateCell].data().dptr_);
    if (default_param.state_outputs && req[rnn_enum::kStateCellOut] != kNullOp) {
      statecellout_mem =
          CreateDNNLMem(outputs[rnn_enum::kStateCellOut], cell_desc, req[rnn_enum::kStateCellOut]);
      dst_state_cell = static_cast<char*>(statecellout_mem.second->get_data_handle());
    }
  }

  if (fwd_inf_vec_.size() == 1) {
    fwd_inf_vec_.front().SetNewDataMem(
        src, src_state, src_state_cell, dst, dst_state, dst_state_cell, data_dtype);
    if (is_training) {
      fwd_trn_vec_.front().FetchData(fwd_inf_vec_.front());
    }
  } else {
    CHECK_EQ(fwd_inf_vec_.size(), dst_.size() + 1) << "Output memory error.";
    size_t state_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ * iter_size *
                         mshadow::mshadow_sizeof(data_dtype);
    size_t cell_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ * state_size *
                        mshadow::mshadow_sizeof(data_dtype);

    // Set input data memory for the first layer. This stores intermediate output
    // results in this->xxx, used as the source input of the next layer.
    fwd_inf_vec_.front().SetNewDataMem(src,
                                       src_state,
                                       src_state_cell,
                                       this->dst_.front()->get_data_handle(),
                                       dst_state,
                                       dst_state_cell,
                                       data_dtype);
    if (is_training) {
      fwd_trn_vec_.front().FetchData(fwd_inf_vec_.front());
    }
    // 1st_lyr -> dst_handle -> next_lyr -> dst_handle -> next_lyr -> ...
    for (size_t lyr = 1; lyr < fwd_inf_vec_.size() - 1; ++lyr) {
      src_state += state_bytes;
      if (dst_state)
        dst_state += state_bytes;
      if (src_state_cell)
        src_state_cell += cell_bytes;
      if (dst_state_cell)
        dst_state_cell += cell_bytes;
      fwd_inf_vec_.at(lyr).SetNewDataMem(this->dst_.at(lyr - 1)->get_data_handle(),
                                         src_state,
                                         src_state_cell,
                                         this->dst_.at(lyr)->get_data_handle(),
                                         dst_state,
                                         dst_state_cell,
                                         data_dtype);
      if (is_training) {
        fwd_trn_vec_.at(lyr).FetchData(fwd_inf_vec_.at(lyr));
      }
    }
    // Set output data memory for the last layer.
    src_state += state_bytes;
    if (dst_state)
      dst_state += state_bytes;
    if (src_state_cell)
      src_state_cell += cell_bytes;
    if (dst_state_cell)
      dst_state_cell += cell_bytes;
    fwd_inf_vec_.back().SetNewDataMem(this->dst_.back()->get_data_handle(),
                                      src_state,
                                      src_state_cell,
                                      dst,
                                      dst_state,
                                      dst_state_cell,
                                      data_dtype);
    if (is_training) {
      fwd_trn_vec_.back().FetchData(fwd_inf_vec_.back());
    }
  }
  if (is_training) {
    for (auto& trn_lyr : fwd_trn_vec_)
      RegisterDNNLRnn(trn_lyr);
  } else {
    for (auto& inf_lyr : fwd_inf_vec_)
      RegisterDNNLRnn(inf_lyr);
  }
  CommitOutput(outputs[rnn_enum::kOut], out_mem);
  if (default_param.state_outputs) {
    CommitOutput(outputs[rnn_enum::kStateOut], stateout_mem);
    if (default_param.mode == rnn_enum::kLstm)
      CommitOutput(outputs[rnn_enum::kStateCellOut], statecellout_mem);
  }
  DNNLStream::Get()->Submit();
}

void DNNLRnnOp::Backward(const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  using tag = dnnl::memory::format_tag;
  TmpMemMgr::Get()->Init(ctx.requested[1]);
  const RNNParam& default_param = full_param_.default_param;
  const int data_dtype          = inputs[rnn_enum::kData].dtype();
  const int w_dtype             = inputs[rnn_enum::kParams].dtype();

  // Initialize the bwd_vec_
  if (bwd_vec_.size() != fwd_inf_vec_.size()) {
    bwd_vec_.clear();
    for (size_t lyr = 0; lyr < fwd_inf_vec_.size(); ++lyr)
      bwd_vec_.emplace_back(
          fwd_trn_vec_.at(lyr), inputs[rnn_enum::kData], inputs[rnn_enum::kParams]);
  }
  // Fetch weights, src and dst from Forward layer
  if (bwd_vec_.size() != fwd_trn_vec_.size())
    LOG(FATAL) << "oneDNN RNN fusion error.";
  for (size_t lyr = 0; lyr < bwd_vec_.size(); ++lyr) {
    bwd_vec_.at(lyr).FetchDataWeightsMem(fwd_trn_vec_.at(lyr));
    bwd_vec_.at(lyr).SetWeightsGradsMem();
  }

  const size_t w_bytes = mshadow::mshadow_sizeof(w_dtype);
  // Get temporary memory for diff_src, diff_state, diff_statecell
  const int num_layers = default_param.num_layers;
  const int seq_length = default_param.seq_length_;
  const int batch_size = default_param.batch_size_;
  const int input_size = default_param.input_size_;
  const int state_size = default_param.state_size;
  const int directions = default_param.bidirectional ? 2 : 1;
  dnnl::memory::desc src_desc(
      {seq_length, batch_size, input_size}, get_dnnl_type(data_dtype), tag::tnc);
  dnnl::memory::desc state_desc(
      {num_layers, directions, batch_size, state_size}, get_dnnl_type(data_dtype), tag::ldnc);
  auto diff_input_mem = CreateDNNLMem(outputs[rnn_enum::kData], src_desc, req[rnn_enum::kData]);
  dnnl_output_t diff_state_mem;
  dnnl_output_t diff_statecell_mem;
  // index description of outputs NDArray
  //   0    1    2     3
  // | dx | dw | dhx | dcx|
  char* dx = static_cast<char*>(diff_input_mem.second->get_data_handle());
  char* dw = static_cast<char*>(outputs[rnn_enum::kParams].data().dptr_);
  char* db = dw + (inputs[rnn_enum::kParams].data().Size() -
                   GetRnnBiasSize(default_param.num_layers,
                                  default_param.state_size,
                                  default_param.bidirectional + 1,
                                  default_param.mode)) *
                      w_bytes;
  diff_state_mem = CreateDNNLMem(outputs[rnn_enum::kState], state_desc, req[rnn_enum::kState]);
  char* dhx      = static_cast<char*>(diff_state_mem.second->get_data_handle());
  char* dcx      = nullptr;
  if (full_param_.default_param.mode == rnn_enum::kLstm && req[rnn_enum::kStateCell] != kNullOp) {
    diff_statecell_mem =
        CreateDNNLMem(outputs[rnn_enum::kStateCell], state_desc, req[rnn_enum::kStateCell]);
    dcx = static_cast<char*>(diff_statecell_mem.second->get_data_handle());
  }

  // index description of inputs NDArray
  //   0   1   2    3   4    5     6    7    8     9
  // | x | w | hx | y | dy | hy | dhy | cx | cy | dcy |
  char* dy  = static_cast<char*>(inputs[4].data().dptr_);
  char* dhy = nullptr;
  if (default_param.state_outputs)
    dhy = static_cast<char*>(inputs[6].data().dptr_);

  char* dcy = nullptr;
  if ((default_param.mode == rnn_enum::kLstm) && default_param.state_outputs)
    dcy = static_cast<char*>(inputs[9].data().dptr_);

  if (bwd_vec_.size() == 1) {
    bwd_vec_.back().SetDataGradsMem(dx, dhx, dcx, dy, dhy, dcy, data_dtype);
    RegisterDNNLRnn(bwd_vec_.back());
  } else {
    const size_t state_bytes = (default_param.bidirectional + 1) * default_param.batch_size_ *
                               default_param.state_size * mshadow::mshadow_sizeof(data_dtype);
    if (diff_src == nullptr) {
      auto desc = dnnl::memory::desc(
          full_param_.layer_params.back().src_dims, get_dnnl_type(data_dtype), tag::tnc);
      diff_src = std::make_shared<dnnl::memory>(desc, CpuEngine::Get()->get_engine());
    }
    // Sets primitives from bottom to top, then submits them in reversed order.
    bwd_vec_.front().SetDataGradsMem(
        dx, dhx, dcx, diff_src->get_data_handle(), dhy, dcy, data_dtype);
    for (size_t lyr = 1; lyr < bwd_vec_.size() - 1; ++lyr) {
      if (dhx)
        dhx += state_bytes;
      if (dcx)
        dcx += state_bytes;
      if (dhy)
        dhy += state_bytes;
      if (dcy)
        dcy += state_bytes;
      bwd_vec_.at(lyr).SetDataGradsMem(
          diff_src->get_data_handle(), dhx, dcx, diff_src->get_data_handle(), dhy, dcy, data_dtype);
    }
    if (dhx)
      dhx += state_bytes;
    if (dcx)
      dcx += state_bytes;
    if (dhy)
      dhy += state_bytes;
    if (dcy)
      dcy += state_bytes;
    bwd_vec_.back().SetDataGradsMem(
        diff_src->get_data_handle(), dhx, dcx, dy, dhy, dcy, data_dtype);

    for (std::vector<DNNLRnnBackward>::const_reverse_iterator bwd = bwd_vec_.rbegin();
         bwd != bwd_vec_.rend();
         ++bwd) {
      RegisterDNNLRnn(*bwd);
    }
  }
  CommitOutput(outputs[rnn_enum::kData], diff_input_mem);
  CommitOutput(outputs[rnn_enum::kState], diff_state_mem);
  if (full_param_.default_param.mode == rnn_enum::kLstm)
    CommitOutput(outputs[rnn_enum::kStateCell], diff_statecell_mem);
  DNNLStream::Get()->Submit();

  // Commit weights diff
  if (req[rnn_enum::kParams] != kNullOp) {
    const int directions = default_param.bidirectional ? 2 : 1;
    for (size_t lyr = 0; lyr < bwd_vec_.size(); ++lyr) {
      bwd_vec_.at(lyr).CommitWeightsGrads(dw, db, req[rnn_enum::kParams], w_dtype);
      dw += full_param_.layer_params.at(lyr).single_w_size * directions * w_bytes;
      db += full_param_.layer_params.at(lyr).native_single_b_size * directions * w_bytes;
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
