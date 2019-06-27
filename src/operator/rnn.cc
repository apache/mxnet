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
 * Copyright (c) 2015 by Contributors
 * \file rnn.cc
 * \brief
 * \author Sebastian Bodenstein
*/

#include <iterator>

#include "./rnn-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(RNNParam);
static inline std::vector<std::string> ListArguments(const RNNParam& param_) {
  // All RNNs start off with same 3 input arguments
  std::vector<std::string> arguments{"data", "parameters", "state"};

  // LSTMs also have an additional state_cell argument
  if (param_.mode == rnn_enum::kLstm) {
    arguments.emplace_back("state_cell");
  }

  // All RNNs have option of additional sequence_length argument
  if (param_.use_sequence_length) {
    arguments.emplace_back("sequence_length");
  }

  return arguments;
}

static bool RNNShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_shape,
                     std::vector<TShape> *out_shape) {
  const RNNParam& param_ = nnvm::get<RNNParam>(attrs.parsed);
  using namespace mshadow;

  // Query param_ object to figure out what the expectd input arguments are
  std::vector<std::string> expected_arguments = ListArguments(param_);

  CHECK_EQ(in_shape->size(), expected_arguments.size()) << "Input shape mismatch. Expected " <<
    expected_arguments.size() << " input parameters but got " << in_shape->size() << ".";

  const TShape &dshape = (*in_shape)[rnn_enum::kData];
  if (!mxnet::ndim_is_known(dshape)) return false;
  CHECK_EQ(dshape.ndim(), 3U) \
      << "Input data should be rank-3 tensor of dim [sequence length, batch size, input size]";
  // data: [sequence len, batch, input dimension]
  int batch_size = dshape[1];
  int input_size = dshape[2];
  int numDirections = param_.bidirectional ? 2 : 1;
  int total_layers = numDirections * param_.num_layers;  // double for bidirectional
  int layer_size = (param_.projection_size.has_value()) ?
      param_.projection_size.value() : param_.state_size;
  SHAPE_ASSIGN_CHECK(*in_shape,
                     rnn_enum::kState,
                     Shape3(total_layers, batch_size, layer_size));
  if (param_.mode == rnn_enum::kLstm) {
    SHAPE_ASSIGN_CHECK(*in_shape,
                       rnn_enum::kStateCell,
                       Shape3(total_layers, batch_size, param_.state_size));
  }

  // calculate parameter vector length
  int param_size = GetRnnParamSize(param_.num_layers,
                                   input_size,
                                   param_.state_size,
                                   numDirections,
                                   param_.mode,
                                   param_.projection_size);
  SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kParams, Shape1(param_size));

  // Check on sequence_length shape if using
  if (param_.use_sequence_length) {
    size_t seq_len_input_idx = rnn_enum::kSequenceLength;
    if (param_.mode != rnn_enum::kLstm) --seq_len_input_idx;

    SHAPE_ASSIGN_CHECK(*in_shape, seq_len_input_idx, Shape1(batch_size));
  }

  out_shape->clear();
  // output: [sequence len, batch, output size]
  TShape oshape = dshape;
  if (param_.projection_size.has_value()) {
    oshape[2] = numDirections * param_.projection_size.value();
  } else {
    oshape[2] = numDirections * param_.state_size;
  }
  out_shape->push_back(oshape);
  if (param_.state_outputs) {
    // outStateShape: [layer_num, batch, state size]
    TShape outStateShape = dshape;
    outStateShape[0] = total_layers;
    outStateShape[1] = batch_size;
    if (param_.projection_size.has_value()) {
      outStateShape[2] = param_.projection_size.value();
    } else {
      outStateShape[2] = param_.state_size;
    }
    out_shape->push_back(outStateShape);
    // Deal with lstm cell state
    if (param_.mode == rnn_enum::kLstm) {
      TShape cellStateShape = dshape;
      cellStateShape[0] = total_layers;
      cellStateShape[1] = batch_size;
      cellStateShape[2] = param_.state_size;
      out_shape->push_back(cellStateShape);
    }
  }

  return true;
}

static bool RNNType(const nnvm::NodeAttrs& attrs,
                    std::vector<int> *in_type,
                    std::vector<int> *out_type) {
  const RNNParam& param_ = nnvm::get<RNNParam>(attrs.parsed);

  CHECK_EQ(in_type->size(), GetNumInputArguments(param_));

  size_t seq_len_input_idx = rnn_enum::kSequenceLength;
  if (param_.mode != rnn_enum::kLstm)  --seq_len_input_idx;

  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  std::vector<std::string> arguments = ListArguments(param_);
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      TYPE_ASSIGN_CHECK(*in_type, i, dtype);
    } else {
      // If using sequence length argument, it has its own indexing type
      // All other input arguments must match the main data type
      if (!(param_.use_sequence_length && i == seq_len_input_idx)) {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, arguments[i]);
      }
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  if (param_.state_outputs) {
    out_type->push_back(dtype);
    // Deal with lstm cell state
    if (param_.mode == rnn_enum::kLstm) {
      out_type->push_back(dtype);
    }
  }
  return true;
}

static std::vector<ResourceRequest> RNNResourceEx(const NodeAttrs& attrs, const int dev_mask,
                                                  const DispatchMode dispatch_mode) {
  std::vector<ResourceRequest> request;
  if (dev_mask == kGPU) {
#if MXNET_USE_CUDNN_RNN
    request.emplace_back(ResourceRequest::kTempSpace);

    const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
    if (param.p != 0 && 1.0f - param.p > 0) {
      request.emplace_back(ResourceRequest::kCuDNNDropoutDesc);
    }
#endif
  }
  return request;
}

inline static bool RNNStorageType(const nnvm::NodeAttrs& attrs,
                                  const int dev_mask,
                                  DispatchMode* dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  DispatchMode wanted_mode = DispatchMode::kFCompute;

  #if MXNET_USE_MKLDNN == 1
    wanted_mode = DispatchMode::kFComputeEx;
  #endif

  return storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                             dispatch_mode, wanted_mode);
}

struct RNNGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr &n,
          const std::vector<nnvm::NodeEntry> &ograd) const {
    const RNNParam& params = nnvm::get<RNNParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> heads{ n->inputs[rnn_enum::kData],
      n->inputs[rnn_enum::kParams], n->inputs[rnn_enum::kState] };
    heads.emplace_back(n, rnn_enum::kOut, 0);
    heads.push_back(ograd[rnn_enum::kOut]);
    if (params.state_outputs) {
      heads.emplace_back(n, rnn_enum::kStateOut, 0);
      heads.push_back(ograd[rnn_enum::kStateOut]);
    }
    if (params.mode == rnn_enum::kLstm) {
      heads.push_back(n->inputs[rnn_enum::kStateCell]);
      if (params.state_outputs) {
        heads.emplace_back(n, rnn_enum::kStateCellOut, 0);
        heads.push_back(ograd[rnn_enum::kStateCellOut]);
      }
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

#if MXNET_USE_MKLDNN == 1
static void RNNStatefulComputeCPU(const OpStatePtr& state_ptr,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  std::vector<TBlob> in_blobs;
  std::vector<TBlob> out_blobs;
  std::vector<NDArray> temp_ndarrays_i;
  std::vector<NDArray> temp_ndarrays_o;
  for (const NDArray& in : inputs) {
    if (in.storage_type() == kDefaultStorage) {
      temp_ndarrays_i.push_back(in.Reorder2Default());
      in_blobs.emplace_back(temp_ndarrays_i.back().data());
    } else {
      in_blobs.emplace_back(in.data());
    }
  }

  for (const NDArray& out : outputs) {
    if (out.storage_type() == kDefaultStorage) {
      temp_ndarrays_o.push_back(out.Reorder2Default());
      out_blobs.emplace_back(temp_ndarrays_o.back().data());
    } else {
      out_blobs.emplace_back(out.data());
    }
  }
  int dtype = in_blobs[rnn_enum::kData].type_flag_;
  int itype = in_blobs[inputs.size()-1].type_flag_;
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(itype, IType, {
      RNNOp<cpu, DType, IType>& op = state_ptr.get_state<RNNOp<cpu, DType, IType>>();
      const RNNParam& param = op.param_;
      int ngates = 0, nstates = 0;
      GetMKLDNNRNNAlgo(param.mode, &ngates, &nstates);
      int D = param.bidirectional ? 2 : 1;
      Tensor<cpu, 3, DType> x = in_blobs[rnn_enum::kData].get<cpu, 3, DType>(s);
      int T = x.shape_[0];
      int N = x.shape_[1];
      int I = x.shape_[2];
      int H = param.state_size;
      int L = param.num_layers;

      const size_t r_size = GetMKLDNNRNNCacheMemorySize(L, D, T, N, I, H, param.mode);
      if (op.init_mem_ && op.reserve_mem_size_ < r_size) {
        Storage::Get()->Free(op.mem_space_);
        op.init_mem_ = false;
      }
      if (!op.init_mem_) {
        op.mem_space_ = Storage::Get()->Alloc(
            r_size * sizeof(DType),
            Context::CPU());
        op.reserve_mem_size_ = r_size;
        op.init_mem_ = true;
        op.has_cache = false;
      }
      if (op.has_cache && op.x_memory.size() == 0) {
        op.has_cache = false;
      }

      DType* workptr = static_cast<DType*>(op.mem_space_.dptr);
      mkldnn::memory::dims src_layer_tz_0 = {T, N, I};
      mkldnn::memory::dims src_layer_tz = {T, N, D * H};
      mkldnn::memory::dims dst_layer_tz = {T, N, D * H};
      auto dst_layer_md = mkldnn::memory::desc(
        { dst_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
      if (op.x_memory.size() == 0) {
        if (D == 1 && I == H) {
          auto user_src_layer_md = mkldnn::memory::desc(
              { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
          auto user_src_layer_memory_n = mkldnn::memory({ user_src_layer_md, cpu_engine });
          op.x_memory.push_back(user_src_layer_memory_n);

          mkldnn::memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
          mkldnn::memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
          mkldnn::memory::dims bias_tz = {L, 1, ngates, H};
          auto user_weight_layer_md = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_weight_iter_md = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_bias_md = mkldnn::memory::desc({ bias_tz },
              mkldnn_dtype, mkldnn::memory::format::ldgo);
          DType* weight_layer_n = workptr;  //  L * I * ngates * H
          auto user_weight_layer_memory_n
              = mkldnn::memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
          op.wx_memory.push_back(user_weight_layer_memory_n);

          DType* weight_iter_n = weight_layer_n + L * I * ngates * H;  //  L * H * ngates * H
          auto user_weight_iter_memory_n
              = mkldnn::memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
          op.wh_memory.push_back(user_weight_iter_memory_n);

          DType* bias_n = weight_iter_n + L * H * ngates * H;  //  L * ngates * H
          auto user_bias_memory_n =
              mkldnn::memory({ user_bias_md, cpu_engine }, bias_n);
          op.bias_memory.push_back(user_bias_memory_n);

          auto wx_md_n = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          DType* wx_n = bias_n + L * ngates * H;  //   L * ngates * I * H
          auto wx_memory_n =
              mkldnn::memory({ wx_md_n, cpu_engine }, wx_n);
          DType* wh_n = wx_n + L * ngates * I * H;  //  L * ngates * H * H
          auto wh_md_n = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          auto wh_memory_n =
              mkldnn::memory({ wh_md_n, cpu_engine }, wh_n);

          op.concat_weight_memory.push_back(wx_memory_n);
          op.concat_weight_memory.push_back(wh_memory_n);
          workptr = wh_n + L * ngates * H * H;

          mkldnn::memory::dims src_iter_tz_n1 = {1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_md_n1 = mkldnn::memory::desc(
              { src_iter_tz_n1 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          for (int l = 0; l < L; l++) {
            DType* src_iter_n1 = workptr;  //  nstates * N * H
            auto src_iter_memory_n1 =
                mkldnn::memory({ src_iter_md_n1, cpu_engine }, src_iter_n1);
            op.concat_iter_memory.push_back(src_iter_memory_n1);
            workptr = src_iter_n1 + nstates * N * H;
          }
          mkldnn::memory::dims src_iter_tz_n = {L, 1, nstates, N, H};  //  ldsnc
          auto src_iter_md_n = mkldnn::memory::desc(
              { src_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* src_iter_n = workptr;  //  L * nstates * N * H
          auto src_iter_memory_n =
              mkldnn::memory({ src_iter_md_n, cpu_engine }, src_iter_n);
          op.concat_iter_memory.push_back(src_iter_memory_n);
          op.hcx_memory.push_back(src_iter_memory_n);
          DType* dst_layer_n = src_iter_n + L * nstates * N * H;  //  T * N * D * H
          auto dst_layer_memory_n
              = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_n);
          op.y_memory.push_back(dst_layer_memory_n);

          mkldnn::memory::dims dst_iter_tz_n = {L, 1, nstates, N, H};  //  ldsnc
          auto dst_iter_md_n = mkldnn::memory::desc(
              { dst_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  L * nstates * N * H
          auto dst_iter_memory_n =
              mkldnn::memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
          op.hcy_memory.push_back(dst_iter_memory_n);
          workptr = dst_iter_n + L * nstates * N * H;

        } else {
          auto user_src_layer_md_0 = mkldnn::memory::desc(
              { src_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::tnc);
          auto user_src_layer_memory_0 = mkldnn::memory({ user_src_layer_md_0, cpu_engine });
          op.x_memory.push_back(user_src_layer_memory_0);

          mkldnn::memory::dims weights_layer_tz_0 = {1, D, I, ngates, H};  //  ldigo
          mkldnn::memory::dims weights_iter_tz_0 = {1, D, H, ngates, H};  //  ldigo
          mkldnn::memory::dims bias_tz_0 = {1, D, ngates, H};
          auto user_weight_layer_md_0 = mkldnn::memory::desc(
              { weights_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_weight_iter_md_0 = mkldnn::memory::desc(
              { weights_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_bias_md_0 = mkldnn::memory::desc({ bias_tz_0 },
              mkldnn_dtype, mkldnn::memory::format::ldgo);

          DType* weight_layer_0 = workptr;  //  D * I * ngates * H
          auto user_weight_layer_memory_0
              = mkldnn::memory({ user_weight_layer_md_0, cpu_engine }, weight_layer_0);
          op.wx_memory.push_back(user_weight_layer_memory_0);

          DType* weight_iter_0 = weight_layer_0 + D * I * ngates * H;  //  D * H * ngates * H
          auto user_weight_iter_memory_0
              = mkldnn::memory({ user_weight_iter_md_0, cpu_engine }, weight_iter_0);
          op.wh_memory.push_back(user_weight_iter_memory_0);

          DType* bias_0 = weight_iter_0 + D * H * ngates * H;  //  D * ngates * H
          auto user_bias_memory_0 =
              mkldnn::memory({ user_bias_md_0, cpu_engine }, bias_0);
          op.bias_memory.push_back(user_bias_memory_0);
          workptr = bias_0 + D * ngates * H;

          auto wx_md_0 = mkldnn::memory::desc(
              { weights_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          auto wx_memory_0 =
              mkldnn::memory({ wx_md_0, cpu_engine });
          auto wh_md_0 = mkldnn::memory::desc(
              { weights_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          auto wh_memory_0 =
              mkldnn::memory({ wh_md_0, cpu_engine });
          if (D == 2) {
            DType* wx_0 = workptr;  //  D * ngates * I * H
            wx_memory_0.set_data_handle(wx_0);
            DType* wh_0 = wx_0 + D * ngates * I * H;  //  D * ngates * H * H
            wh_memory_0.set_data_handle(wh_0);
            workptr = wh_0 + D * ngates * H * H;
          }
          op.concat_weight_memory.push_back(wx_memory_0);
          op.concat_weight_memory.push_back(wh_memory_0);

          mkldnn::memory::dims src_iter_undi_tz_0 = {1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_undi_md_0 = mkldnn::memory::desc(
              { src_iter_undi_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* src_iter_undi_0 = workptr;  //  nstates * N * H
          auto src_iter_undi_memory_0 =
              mkldnn::memory({ src_iter_undi_md_0, cpu_engine }, src_iter_undi_0);
          op.concat_iter_memory.push_back(src_iter_undi_memory_0);
          workptr = src_iter_undi_0 + nstates * N * H;
          if (D == 1) {
            op.hcx_memory.push_back(src_iter_undi_memory_0);
          } else {
            DType* src_iter_undi2_0 = workptr;  //  nstates * N * H
            auto src_iter_undi2_memory_0 =
                mkldnn::memory({ src_iter_undi_md_0, cpu_engine }, src_iter_undi2_0);
            op.concat_iter_memory.push_back(src_iter_undi2_memory_0);

            mkldnn::memory::dims src_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
            auto src_iter_md_0 = mkldnn::memory::desc(
                { src_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            DType* src_iter_0 = src_iter_undi2_0 + nstates * N * H;  //  D * nstates * N * H
            auto src_iter_memory_0 =
                mkldnn::memory({ src_iter_md_0, cpu_engine }, src_iter_0);
            op.concat_iter_memory.push_back(src_iter_memory_0);
            op.hcx_memory.push_back(src_iter_memory_0);
            workptr = src_iter_0 + D * nstates * N * H;
          }

          DType* dst_layer_0 = workptr;  //  T * N * D * H
          auto dst_layer_memory_0
              = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_0);
          op.y_memory.push_back(dst_layer_memory_0);

          mkldnn::memory::dims dst_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
          auto dst_iter_md_0 = mkldnn::memory::desc(
              { dst_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* dst_iter_0 = dst_layer_0 + T * N * D * H;  //  D * nstates * N * H
          auto dst_iter_memory_0 =
              mkldnn::memory({ dst_iter_md_0, cpu_engine }, dst_iter_0);
          op.hcy_memory.push_back(dst_iter_memory_0);
          workptr = dst_iter_0 + D * nstates * N * H;

          //  next L - 1 layers
          if (L > 1 && D == 1) {
            auto user_src_layer_md = mkldnn::memory::desc(
                { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
            auto user_src_layer_memory = mkldnn::memory({ user_src_layer_md, cpu_engine });
            op.x_memory.push_back(user_src_layer_memory);

            mkldnn::memory::dims weights_layer_tz = {L - 1, 1, H, ngates, H};  //  ldigo
            mkldnn::memory::dims weights_iter_tz = {L - 1, 1, H, ngates, H};  //  ldigo
            mkldnn::memory::dims bias_tz = {L - 1, 1, ngates, H};
            auto user_weight_layer_md = mkldnn::memory::desc(
                { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
            auto user_weight_iter_md = mkldnn::memory::desc(
                { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
            auto user_bias_md = mkldnn::memory::desc({ bias_tz },
                mkldnn_dtype, mkldnn::memory::format::ldgo);

            DType* weight_layer_n = workptr;  //  (L - 1) * H * ngates * H
            auto user_weight_layer_memory_n
                = mkldnn::memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
            op.wx_memory.push_back(user_weight_layer_memory_n);

            DType* weight_iter_n = weight_layer_n +
                (L - 1) * H * ngates * H;  //  (L - 1) * H * ngates * H
            auto user_weight_iter_memory_n
                = mkldnn::memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
            op.wh_memory.push_back(user_weight_iter_memory_n);

            DType* bias_n = weight_iter_n + (L - 1) * H * ngates * H;  //  (L - 1) * ngates * H
            auto user_bias_memory_n =
                mkldnn::memory({ user_bias_md, cpu_engine }, bias_n);
            op.bias_memory.push_back(user_bias_memory_n);

            auto wx_md_n = mkldnn::memory::desc(
                { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
            DType* wx_n = bias_n + (L - 1) * ngates * H;  //  (L - 1) * ngates * H * H
            auto wx_memory_n =
                mkldnn::memory({ wx_md_n, cpu_engine }, wx_n);
            DType* wh_n = wx_n + (L - 1) * ngates * H * H;  //  (L - 1) * ngates * H * H
            auto wh_md_n = mkldnn::memory::desc(
                { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
            auto wh_memory_n =
                mkldnn::memory({ wh_md_n, cpu_engine }, wh_n);

            op.concat_weight_memory.push_back(wx_memory_n);
            op.concat_weight_memory.push_back(wh_memory_n);
            workptr = wh_n + (L - 1) * ngates * H * H;

            mkldnn::memory::dims src_iter_tz_n1 = {1, 1, nstates, N, H};  //  ldsnc
            auto src_iter_md_n1 = mkldnn::memory::desc(
                { src_iter_tz_n1 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            for (int l = 0; l < L - 1; l++) {
              DType* src_iter_n1 = workptr;  //  nstates * N * H
              auto src_iter_memory_n1 =
                  mkldnn::memory({ src_iter_md_n1, cpu_engine }, src_iter_n1);
              op.concat_iter_memory.push_back(src_iter_memory_n1);
              workptr = src_iter_n1 + nstates * N * H;
            }
            mkldnn::memory::dims src_iter_tz_n = {L - 1, 1, nstates, N, H};  //  ldsnc
            auto src_iter_md_n = mkldnn::memory::desc(
                { src_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            DType* src_iter_n = workptr;  //  (L - 1) * nstates * N * H
            auto src_iter_memory_n =
                mkldnn::memory({ src_iter_md_n, cpu_engine }, src_iter_n);
            op.concat_iter_memory.push_back(src_iter_memory_n);
            op.hcx_memory.push_back(src_iter_memory_n);

            DType* dst_layer_n = src_iter_n + (L - 1) * nstates * N * H;  //  T * N * D * H
            auto dst_layer_memory_n
                = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_n);
            op.y_memory.push_back(dst_layer_memory_n);

            mkldnn::memory::dims dst_iter_tz_n = {L - 1, 1, nstates, N, H};  //  ldsnc
            auto dst_iter_md_n = mkldnn::memory::desc(
                { dst_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  (L - 1) * nstates * N * H
            auto dst_iter_memory_n =
                mkldnn::memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
            op.hcy_memory.push_back(dst_iter_memory_n);
          }

          if (L > 1 && D == 2) {
            mkldnn::memory::dims weights_layer_tz = {1, D, H * D, ngates, H};  //  ldigo
            mkldnn::memory::dims weights_iter_tz = {1, D, H, ngates, H};  //  ldigo
            mkldnn::memory::dims bias_tz = {1, D, ngates, H};
            auto user_weight_layer_md = mkldnn::memory::desc(
                { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
            auto user_weight_iter_md = mkldnn::memory::desc(
                { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
            auto user_bias_md = mkldnn::memory::desc({ bias_tz },
                mkldnn_dtype, mkldnn::memory::format::ldgo);

            auto user_src_layer_md = mkldnn::memory::desc(
                { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
            auto user_src_layer_memory = mkldnn::memory({ user_src_layer_md, cpu_engine });
            op.x_memory.push_back(user_src_layer_memory);

            auto wx_md_n = mkldnn::memory::desc(
                { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
            auto wh_md_n = mkldnn::memory::desc(
                { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);

            for (int l = 0; l < L; l++) {
              DType* weight_layer_n = workptr;  //  D * (H * D) * ngates * H
              auto user_weight_layer_memory_n
                  = mkldnn::memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
              op.wx_memory.push_back(user_weight_layer_memory_n);

              DType* weight_iter_n = weight_layer_n +
                  D * (H * D) * ngates * H;  //  D * H * ngates * H
              auto user_weight_iter_memory_n
                  = mkldnn::memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
              op.wh_memory.push_back(user_weight_iter_memory_n);

              DType* bias_n = weight_iter_n + D * H * ngates * H;  //  D * ngates * H
              auto user_bias_memory_n =
                  mkldnn::memory({ user_bias_md, cpu_engine }, bias_n);
              op.bias_memory.push_back(user_bias_memory_n);
              workptr = bias_n + D * ngates * H;
            }

            DType* wx_n = workptr;  //  D * ngates * (D * H) * H
            DType* wh_n = wx_n + D * ngates * (D * H) * H;  //  D * ngates * H * H
            auto wx_memory_n =
                mkldnn::memory({ wx_md_n, cpu_engine }, wx_n);
            auto wh_memory_n =
                mkldnn::memory({ wh_md_n, cpu_engine }, wh_n);
            op.concat_weight_memory.push_back(wx_memory_n);
            op.concat_weight_memory.push_back(wh_memory_n);

            mkldnn::memory::dims src_iter_undi_tz = {1, 1, nstates, N, H};  //  ldsnc
            auto src_iter_undi_md = mkldnn::memory::desc(
                { src_iter_undi_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            DType* src_iter_undi = wh_n + D * ngates * H * H;  //  nstates * N * H
            auto src_iter_undi_memory =
                mkldnn::memory({ src_iter_undi_md, cpu_engine }, src_iter_undi);
            op.concat_iter_memory.push_back(src_iter_undi_memory_0);

            DType* src_iter_undi2 = src_iter_undi + nstates * N * H;  //  nstates * N * H
            auto src_iter_undi2_memory =
                mkldnn::memory({ src_iter_undi_md, cpu_engine }, src_iter_undi2);
            op.concat_iter_memory.push_back(src_iter_undi2_memory);

            mkldnn::memory::dims src_iter_tz = {1, D, nstates, N, H};  //  ldsnc
            auto src_iter_md = mkldnn::memory::desc(
                { src_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            DType* src_iter = src_iter_undi2 + nstates * N * H;  //  D * nstates * N * H
            auto src_iter_memory =
                mkldnn::memory({ src_iter_md, cpu_engine }, src_iter);
            op.concat_iter_memory.push_back(src_iter_memory);
            op.hcx_memory.push_back(src_iter_memory);

            DType* dst_layer_n = src_iter + D * nstates * N * H;  //  T * N * D * H
            auto dst_layer_memory_n
                = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_n);
            op.y_memory.push_back(dst_layer_memory_n);

            mkldnn::memory::dims dst_iter_tz_n = {1, D, nstates, N, H};  //  ldsnc
            auto dst_iter_md_n = mkldnn::memory::desc(
                { dst_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
            DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  D * nstates * N * H
            auto dst_iter_memory_n =
                mkldnn::memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
            op.hcy_memory.push_back(dst_iter_memory_n);
          }
        }
      }
      op.Forward(ctx, in_blobs, req, out_blobs);
    });
  });
}
#endif

NNVM_REGISTER_OP(RNN)
.add_alias("_npx_rnn")
.describe(R"code(Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
implemented, with both multi-layer and bidirectional support.

When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

**Vanilla RNN**

Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported:
ReLU and Tanh.

With ReLU activation function:

.. math::
    h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

With Tanh activtion function:

.. math::
    h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

Reference paper: Finding structure in time - Elman, 1988.
https://crl.ucsd.edu/~elman/Papers/fsit.pdf

**LSTM**

Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf

.. math::
  \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array}

**GRU**

Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078

The definition of GRU here is slightly different from paper but compatible with CUDNN.

.. math::
  \begin{array}{ll}
            r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
            \end{array}
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<RNNParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
  return GetNumInputArguments(params);
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
  //  kOut
  int num_outputs = 1;
  if (params.state_outputs) {
    // kOut, kStateOut, kStateCellOut
    num_outputs = (params.mode == rnn_enum::kLstm) ? 3 : 2;
  }

  return num_outputs;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
  const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
  return ListArguments(params);
})
.set_attr<mxnet::FInferShape>("FInferShape", RNNShape)
.set_attr<nnvm::FInferType>("FInferType", RNNType)
.set_attr<FInferStorageType>("FInferStorageType", RNNStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateRNNState)
.set_attr<FStatefulCompute>("FStatefulCompute<cpu>", RNNStatefulCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", RNNStatefulComputeCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", RNNGrad{"_backward_RNN"})
.set_attr<FResourceRequestEx>("FResourceRequestEx", RNNResourceEx)
.add_argument("data", "NDArray-or-Symbol", "Input data to RNN")
.add_argument("parameters", "NDArray-or-Symbol",
              "Vector of all RNN trainable parameters concatenated")
.add_argument("state", "NDArray-or-Symbol", "initial hidden state of the RNN")
.add_argument("state_cell", "NDArray-or-Symbol",
              "initial cell state for LSTM networks (only for LSTM)")
.add_argument("sequence_length", "NDArray-or-Symbol",
              "Vector of valid sequence lengths for each element in batch. (Only used if"
              " use_sequence_length kwarg is True)")
.add_arguments(RNNParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_RNN)
.set_num_outputs([](const NodeAttrs& attrs) {
  const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
  return GetNumInputArguments(params);
})
.set_attr_parser(ParamParser<RNNParam>)
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FStatefulCompute>("FStatefulCompute<cpu>", RNNStatefulGradCompute<cpu>)
.set_attr<FResourceRequestEx>("FResourceRequestEx", RNNResourceEx);
}  // namespace op
}  // namespace mxnet
