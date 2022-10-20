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
 * \file rnn-inl.h
 * \brief
 * \author Sebastian Bodenstein, Shu Zhang
 */
#ifndef MXNET_OPERATOR_RNN_INL_H_
#define MXNET_OPERATOR_RNN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/storage.h>
#include <algorithm>
#include <random>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include "./rnn_impl.h"
#include "../profiler/storage_profiler.h"

#if MXNET_USE_CUDNN == 1
STATIC_ASSERT_CUDNN_VERSION_GE(7000);
#endif
#define MXNET_USE_CUDNN_GE_7200 MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7200

namespace mxnet {
namespace op {

namespace rnn_enum {
enum RNNOpInputs { kData, kParams, kState, kStateCell, kSequenceLength };
enum RNNOpOutputs { kOut, kStateOut, kStateCellOut };
enum RNNModeType { kRnnRelu, kRnnTanh, kLstm, kGru };
enum RNNOpResource { kTempSpace, kCuDNNDropoutDescSpace };
}  // namespace rnn_enum

struct RNNParam : public dmlc::Parameter<RNNParam> {
  uint32_t state_size;
  uint32_t num_layers;
  bool bidirectional, state_outputs;
  int mode;
  float p;
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  index_t seq_length_, batch_size_, input_size_;
#pragma GCC diagnostic pop

  bool use_sequence_length;
  dmlc::optional<int> projection_size;
  dmlc::optional<double> lstm_state_clip_min, lstm_state_clip_max;
  bool lstm_state_clip_nan;

  DMLC_DECLARE_PARAMETER(RNNParam) {
    DMLC_DECLARE_FIELD(state_size).describe("size of the state for each layer");

    DMLC_DECLARE_FIELD(num_layers).describe("number of stacked layers");

    DMLC_DECLARE_FIELD(bidirectional)
        .set_default(false)
        .describe("whether to use bidirectional recurrent layers");

    DMLC_DECLARE_FIELD(mode)
        .add_enum("rnn_relu", rnn_enum::kRnnRelu)
        .add_enum("rnn_tanh", rnn_enum::kRnnTanh)
        .add_enum("lstm", rnn_enum::kLstm)
        .add_enum("gru", rnn_enum::kGru)
        .describe("the type of RNN to compute");

    DMLC_DECLARE_FIELD(p).set_default(0.).set_range(0, 1).describe(
        "drop rate of the dropout on the outputs of each RNN layer, except the last layer.");

    DMLC_DECLARE_FIELD(state_outputs)
        .set_default(false)
        .describe("Whether to have the states as symbol outputs.");

    DMLC_DECLARE_FIELD(projection_size)
        .set_default(dmlc::optional<int>())
        .describe("size of project size");

    DMLC_DECLARE_FIELD(lstm_state_clip_min)
        .set_default(dmlc::optional<double>())
        .describe(
            "Minimum clip value of LSTM states. This option must be used together with "
            "lstm_state_clip_max.");

    DMLC_DECLARE_FIELD(lstm_state_clip_max)
        .set_default(dmlc::optional<double>())
        .describe(
            "Maximum clip value of LSTM states. This option must be used together with "
            "lstm_state_clip_min.");

    DMLC_DECLARE_FIELD(lstm_state_clip_nan)
        .set_default(false)
        .describe(
            "Whether to stop NaN from propagating in state by clipping it to min/max. "
            "If clipping range is not specified, this option is ignored.");

    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter "
            "`sequence_length` "
            "to specify variable length sequence");
  }
  std::string ComputeMode2String(int mode) {
    switch (mode) {
      case rnn_enum::kRnnRelu:
        return "rnn_relu";
      case rnn_enum::kRnnTanh:
        return "rnn_tanh";
      case rnn_enum::kLstm:
        return "lstm";
      case rnn_enum::kGru:
        return "gru";
      default:
        LOG(FATAL) << "Unknown mode enum " << mode;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream state_size_s, num_layers_s, bidirectional_s, state_outputs_s, mode_s, p_s,
        use_sequence_length_s, projection_size_s, lstm_state_clip_min_s, lstm_state_clip_max_s,
        lstm_state_clip_nan_s;
    state_size_s << state_size;
    num_layers_s << num_layers;
    bidirectional_s << bidirectional;
    state_outputs_s << state_outputs;
    mode_s << mode;
    p_s << p;
    use_sequence_length_s << use_sequence_length;
    projection_size_s << projection_size;
    lstm_state_clip_min_s << lstm_state_clip_min;
    lstm_state_clip_max_s << lstm_state_clip_max;
    lstm_state_clip_nan_s << lstm_state_clip_nan;
    (*dict)["state_size"]          = state_size_s.str();
    (*dict)["num_layers"]          = num_layers_s.str();
    (*dict)["bidirectional"]       = bidirectional_s.str();
    (*dict)["state_outputs"]       = state_outputs_s.str();
    (*dict)["mode"]                = ComputeMode2String(mode);
    (*dict)["p"]                   = p_s.str();
    (*dict)["use_sequence_length"] = use_sequence_length_s.str();
    (*dict)["projection_size"]     = projection_size_s.str();
    (*dict)["lstm_state_clip_min"] = lstm_state_clip_min_s.str();
    (*dict)["lstm_state_clip_max"] = lstm_state_clip_max_s.str();
    (*dict)["lstm_state_clip_nan"] = lstm_state_clip_nan_s.str();
  }
};

inline index_t GetRnnParamSize(int num_layer,
                               index_t input_size,
                               int state_size,
                               int direction,
                               int mode,
                               const dmlc::optional<int>& projection_size) {
  int size = state_size * direction;
  switch (mode) {
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      size *= 4;
      break;
    case rnn_enum::kGru:
      size *= 3;
      break;
  }
  index_t size1 = (input_size + state_size + 2) * size;              // first layer size
  index_t size2 = (state_size * direction + state_size + 2) * size;  // other layers size
  if (projection_size.has_value()) {
    index_t proj_size = projection_size.value();
    size1             = (input_size + proj_size + 2) * size;
    size2             = (proj_size * direction + proj_size + 2) * size;
  }
  index_t param_size = size1 + (num_layer - 1) * size2;
  if (projection_size.has_value()) {
    param_size += projection_size.value() * state_size * num_layer * direction;
  }
  return param_size;
}

inline int GetRnnBiasSize(int num_layer, int state_size, int direction, int mode) {
  int size = 2 * state_size * direction * num_layer;
  switch (mode) {
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      size *= 4;
      break;
    case rnn_enum::kGru:
      size *= 3;
      break;
  }
  return size;
}

/*
 * Calculate the space size of the intermediate results for RNN inference.
 * The inference procedure of a fusion RNN operator calculates the outputs
 * layer by layer. In one layer calculation, the steps are:
 *  - wx[1...Ngates] * x[1...T] among all time stamp(sz: TxNxHxNgates)
 *  - wh[1...Ngates] * h[t] time by time(sz: NxHxNgates)
 *  - output -> h[t](, c[t] additionally with Lstm) time by time(sz: NxH(x2))
 *  - intermediate y[1...T] as next layer's inputs(sz: TxNxHxD)
 */
inline size_t GetRNNWorkspaceSize(index_t seq_length,
                                  index_t batch_size,
                                  int hidden_size,
                                  int projection_size,
                                  int direction,
                                  int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = seq_length * batch_size * hidden_size * (4 + direction) +  // wx*x + inter-y
             batch_size * hidden_size * 6 +                             // wh*h + h + c
             seq_length * hidden_size * 8 +  // Used in Backward, Δbx, Δbh
             // temporary dy in backward computation for bidirectional layers
             seq_length * batch_size * hidden_size * (direction - 1 ? direction : 0);
      break;
    case rnn_enum::kGru:
      // Differs with Lstm, the outputs of three gates are also held in memory
      size = seq_length * batch_size * hidden_size * direction * (3 + 1) +  // wx*x + inter-y
             batch_size * hidden_size * (6 + direction);                    // wh*h + h + Ngates
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      size = seq_length * batch_size * hidden_size * direction * 2 +  // wx*x + inter-y
             batch_size * hidden_size * (1 + direction);              // h + Ngates
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

inline size_t GetRNNReserveSpaceSize(int num_layer,
                                     int direction,
                                     index_t seq_length,
                                     index_t batch_size,
                                     int hidden_size,
                                     int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = direction * seq_length * batch_size * hidden_size * (num_layer * 7 - 1);
      break;
    case rnn_enum::kGru:
      size = seq_length * batch_size * hidden_size * direction * (num_layer * 9 - 1) +
             batch_size * hidden_size * direction * 9 + hidden_size * seq_length * 6 +
             seq_length * batch_size * 7 * hidden_size * direction;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      size = seq_length * batch_size * hidden_size * direction * (num_layer * 6 - 1) +
             batch_size * hidden_size * direction * 3 + hidden_size * seq_length * 2 +
             seq_length * batch_size * 2 * hidden_size * direction;
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

inline size_t GetRnnNumInputs(RNNParam param) {
  size_t num_inputs = (param.mode == rnn_enum::kLstm) ? 4U : 3U;
  if (param.use_sequence_length)
    num_inputs += 1U;
  return num_inputs;
}

/**
 * @params: ws: Temp workspace for gemm's output storage.
 *          rs: Reserve space of forward intermediate data used for training.
 *          num_layers: The number of recurrent layers.
 *          direction: direction is 2 if use bidirectional recurrent layers, else is 1;
 *          seq_length: The number of iterations to unroll over.
 *          batch_size: size of batch.
 *          input_size: The number of expected input features.
 *          state_size: The number of hidden state features.
 *          x_ptr: Pointer of tensor x containing the features of the input sequence.
 *                 x's shape is [seq_length, batch_size, input_size]
 *          hx_ptr: Pointer of tensor hx containing the initial hidden state.
 *                  hx's shape is [num_layers, batch_size, state_size]
 *          cx_ptr: Only used in lstm mode. pointer of tensor cx containing the initial cell state.
 *                  cx's shape is [num_layers, batch_size, state_size]
 *          w_ptr: Pointer of tensor w containing weights.
 *          b_ptr: Pointer of tensor w containing bias.
 *          y_ptr: Pointer of tensor y containing the features of the output features from the
 *                 last layers of the RNN. y's shape is [seq_length, batch_size, state_size]
 *          hy_ptr: Pointer of tensor hy containing the hidden state for t=seq_length.
 *                  hy's shape is [num_layers, batch_size, state_size]
 *          cy_ptr: Only used in lstm mode. pointer of tensor cy  containing the cell state
 *                  for t=seq_length. cy' shape is [num_layers, batch_size, state_size]
 *          dropout: should be 0 <= dropout < 1
 *          mode: Specifies the type of RNN to compute.
 */
template <typename DType>
void RNNForwardTraining(DType* ws,
                        DType* rs,
                        bool state_outputs,
                        const int num_layers,
                        const int direction,
                        const index_t seq_length,
                        const index_t batch_size,
                        const index_t input_size,
                        const int state_size,
                        DType* x_ptr,
                        DType* hx_ptr,
                        DType* cx_ptr,
                        DType* w_ptr,
                        DType* b_ptr,
                        DType* y_ptr,
                        DType* hy_ptr,
                        DType* cy_ptr,
                        const float dropout,
                        int mode,
                        std::mt19937& rnd_engine) {  // NOLINT(runtime/references)
  switch (mode) {
    case rnn_enum::kLstm:
      LstmForwardTraining<DType>(ws,
                                 rs,
                                 state_outputs,
                                 num_layers,
                                 direction,
                                 seq_length,
                                 batch_size,
                                 input_size,
                                 state_size,
                                 x_ptr,
                                 hx_ptr,
                                 cx_ptr,
                                 w_ptr,
                                 b_ptr,
                                 y_ptr,
                                 hy_ptr,
                                 cy_ptr,
                                 dropout,
                                 rnd_engine);
      break;
    case rnn_enum::kGru:
      GruForwardTraining<DType>(ws,
                                rs,
                                state_outputs,
                                num_layers,
                                direction,
                                seq_length,
                                batch_size,
                                input_size,
                                state_size,
                                x_ptr,
                                hx_ptr,
                                w_ptr,
                                y_ptr,
                                hy_ptr,
                                dropout,
                                rnd_engine);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNForwardTraining<DType>(ws,
                                       rs,
                                       state_outputs,
                                       num_layers,
                                       direction,
                                       seq_length,
                                       batch_size,
                                       input_size,
                                       state_size,
                                       x_ptr,
                                       hx_ptr,
                                       w_ptr,
                                       y_ptr,
                                       hy_ptr,
                                       dropout,
                                       mode,
                                       rnd_engine);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
}

template <typename DType>
void RNNForwardInference(DType* ws,
                         bool state_outputs,
                         const int num_layers,
                         const int direction,
                         const index_t seq_length,
                         const index_t batch_size,
                         const index_t input_size,
                         const int state_size,
                         const int projection_size,
                         DType* x_ptr,
                         DType* hx_ptr,
                         DType* cx_ptr,
                         DType* w_ptr,
                         DType* b_ptr,
                         DType* y_ptr,
                         DType* hy_ptr,
                         DType* cy_ptr,
                         int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      LstmForwardInference<DType>(ws,
                                  state_outputs,
                                  num_layers,
                                  direction,
                                  seq_length,
                                  batch_size,
                                  input_size,
                                  state_size,
                                  projection_size,
                                  x_ptr,
                                  hx_ptr,
                                  cx_ptr,
                                  w_ptr,
                                  b_ptr,
                                  y_ptr,
                                  hy_ptr,
                                  cy_ptr);
      break;
    case rnn_enum::kGru:
      GruForwardInference<DType>(ws,
                                 state_outputs,
                                 num_layers,
                                 direction,
                                 seq_length,
                                 batch_size,
                                 input_size,
                                 state_size,
                                 x_ptr,
                                 hx_ptr,
                                 w_ptr,
                                 y_ptr,
                                 hy_ptr);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNForwardInference<DType>(ws,
                                        state_outputs,
                                        num_layers,
                                        direction,
                                        seq_length,
                                        batch_size,
                                        input_size,
                                        state_size,
                                        x_ptr,
                                        hx_ptr,
                                        w_ptr,
                                        y_ptr,
                                        hy_ptr,
                                        mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template <typename DType>
void RNNBackward(DType* ws,
                 DType* rs,
                 const int num_layers,
                 const int direction,
                 const index_t seq_length,
                 const index_t batch_size,
                 const index_t input_size,
                 const int state_size,
                 DType* x_ptr,
                 DType* hx_ptr,
                 DType* cx_ptr,
                 DType* w_ptr,
                 DType* y_ptr,
                 DType* dy_ptr,
                 DType* dhy_ptr,
                 DType* dcy_ptr,
                 DType* dx_ptr,
                 DType* dhx_ptr,
                 DType* dcx_ptr,
                 DType* dw_ptr,
                 DType* db_ptr,
                 int req_data,
                 int req_params,
                 int req_state,
                 int req_statecell,
                 const float dropout,
                 int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      LstmBackward<DType>(ws,
                          rs,
                          num_layers,
                          direction,
                          seq_length,
                          batch_size,
                          input_size,
                          state_size,
                          x_ptr,
                          hx_ptr,
                          cx_ptr,
                          w_ptr,
                          y_ptr,
                          dy_ptr,
                          dhy_ptr,
                          dcy_ptr,
                          dx_ptr,
                          dhx_ptr,
                          dcx_ptr,
                          dw_ptr,
                          db_ptr,
                          req_data,
                          req_params,
                          req_state,
                          req_statecell,
                          dropout);
      break;
    case rnn_enum::kGru:
      GruBackward<DType>(ws,
                         rs,
                         num_layers,
                         direction,
                         seq_length,
                         batch_size,
                         input_size,
                         state_size,
                         x_ptr,
                         hx_ptr,
                         w_ptr,
                         dy_ptr,
                         dhy_ptr,
                         dx_ptr,
                         dhx_ptr,
                         dw_ptr,
                         req_data,
                         req_params,
                         req_state,
                         dropout);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNBackward<DType>(ws,
                                rs,
                                num_layers,
                                direction,
                                seq_length,
                                batch_size,
                                input_size,
                                state_size,
                                x_ptr,
                                hx_ptr,
                                w_ptr,
                                dy_ptr,
                                dhy_ptr,
                                dx_ptr,
                                dhx_ptr,
                                dw_ptr,
                                req_data,
                                req_params,
                                req_state,
                                dropout,
                                mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template <typename xpu, typename DType, typename IType>
class RNNOp {
 public:
  RNNParam param_;
  Context ctx_;

  explicit RNNOp(RNNParam param, Context ctx) {
    this->param_ = param;
    this->ctx_   = ctx;

    if (ctx_.dev_type == kGPU) {
#if MXNET_USE_CUDNN == 1
      init_cudnn_ = false;
      dtype_      = mshadow::DataType<DType>::kCudnnFlag;
      // TensorCore algos only allowed on fp16-I/O convolutions if permitted by the global policy.
      // No tests in place for fp16 RNNs, so leave TensorCore disabled for now.
      cudnn_tensor_core_ = false;
      // When fp16 RNN tests are introduced, we can enable TensorCore as follows:
      // cudnn_tensor_core =
      //     mshadow::DataType<DType>::kFlag == mshadow::kFloat16 && GetEnvAllowTensorCore();
      // Defaults
      input_mode_ = CUDNN_LINEAR_INPUT;  // Don't support this yet
      // RNN Mode
      switch (param_.mode) {
        case rnn_enum::kRnnRelu:
          mode_ = CUDNN_RNN_RELU;
          break;
        case rnn_enum::kRnnTanh:
          mode_ = CUDNN_RNN_TANH;
          break;
        case rnn_enum::kLstm:
          mode_ = CUDNN_LSTM;
          break;
        case rnn_enum::kGru:
          mode_ = CUDNN_GRU;
          break;
        default:
          LOG(FATAL) << "Not implmented";
      }
#if MXNET_USE_CUDNN_GE_7200
      if (param_.projection_size.has_value()) {
        CHECK_EQ(param_.mode, rnn_enum::kLstm) << "Projection is only supported for LSTM.";
        CHECK_GE(param_.state_size, param_.projection_size.value())
            << "State size must be larger than projection size.";
      }
#else
      CHECK(!param_.projection_size.has_value())
          << "Projection is only supported for LSTM with CuDNN version later than 7.1.1.";
#endif  // MXNET_USE_CUDNN_GE_7200
#if MXNET_USE_CUDNN_GE_7200
      if (param_.lstm_state_clip_min.has_value() || param_.lstm_state_clip_max.has_value()) {
        CHECK_EQ(param_.mode, rnn_enum::kLstm) << "State clipping is only supported for LSTM.";
        CHECK(param_.lstm_state_clip_min.has_value() && param_.lstm_state_clip_max.has_value())
            << "lstm_state_clip_min and lstm_state_clip_max must be specified together.";
        CHECK_GE(param_.lstm_state_clip_max.value(), param_.lstm_state_clip_min.value())
            << "lstm_state_clip_max must be greater or equal to lstm_state_clip_min";
      }
#else
      CHECK(!param_.lstm_state_clip_min.has_value() && !param_.lstm_state_clip_max.has_value())
          << "State clipping is only supported for LSTM with CuDNN version later than 7.2.1.";
#endif  // MXNET_USE_CUDNN_GE_7200
      // RNN Direction
      direction_ = param_.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
      // Create descriptors
      CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&cx_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&hy_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&cy_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dhx_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dcx_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dhy_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dcy_desc_));

      CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_));
      CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc_));

      CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_));
      CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_));

#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnCreateRNNDataDescriptor(&x_data_desc_));
      CUDNN_CALL(cudnnCreateRNNDataDescriptor(&y_data_desc_));
      CUDNN_CALL(cudnnCreateRNNDataDescriptor(&dx_data_desc_));
      CUDNN_CALL(cudnnCreateRNNDataDescriptor(&dy_data_desc_));
#endif  // MXNET_USE_CUDNN_GE_7200
#else
      if (ctx_.dev_type == kGPU) {
        LOG(FATAL) << "RNN on GPU is only available for cuDNN at the moment.";
      }
#endif  // MXNET_USE_CUDNN == 1
    }

    if (ctx_.dev_type == kCPU) {
      this->init_space_             = false;
      this->temp_init_space_        = false;
      this->reserve_cpu_space_size_ = 0;
      this->temp_cpu_space_size_    = 0;

      if (param_.lstm_state_clip_min.has_value() || param_.lstm_state_clip_max.has_value()) {
        LOG(FATAL) << "LSTM state clipping is only supported for GPU with CuDNN later than 7.2.1";
      }
    }
  }

  ~RNNOp() {
    if (ctx_.dev_type == kGPU) {
#if MXNET_USE_CUDNN == 1
      CUDNN_CALL(cudnnDestroyTensorDescriptor(hx_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(cx_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(hy_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(cy_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dhx_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dcx_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dhy_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dcy_desc_));

      CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc_));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(dw_desc_));
      CUDNN_CALL(cudnnDestroyRNNDescriptor(rnn_desc_));
      CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));
      if (dgrad_sync_event_created_)
        CUDA_CALL(cudaEventDestroy(dgrad_sync_event_));

      if (init_cudnn_) {
        for (size_t i = 0; i < x_desc_vec_.size(); ++i) {
          CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_vec_[i]));
          CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_vec_[i]));
          CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc_vec_[i]));
          CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc_vec_[i]));
        }
        init_cudnn_ = false;
        Storage::Get()->Free(reserve_space_);
      }
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnDestroyRNNDataDescriptor(x_data_desc_));
      CUDNN_CALL(cudnnDestroyRNNDataDescriptor(y_data_desc_));
      CUDNN_CALL(cudnnDestroyRNNDataDescriptor(dx_data_desc_));
      CUDNN_CALL(cudnnDestroyRNNDataDescriptor(dy_data_desc_));
#endif  // MXNET_USE_CUDNN_GE_7200
#endif  // MXNET_USE_CUDNN
    }
  }

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";
    size_t num_inputs = GetRnnNumInputs(param_);

    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    }

    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // get input + output tensors
    Tensor<xpu, 3, DType> x  = in_data[rnn_enum::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 1, DType> w  = in_data[rnn_enum::kParams].get<xpu, 1, DType>(s);
    Tensor<xpu, 3, DType> hx = in_data[rnn_enum::kState].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> y  = out_data[rnn_enum::kOut].get<xpu, 3, DType>(s);

    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    const int direction = param_.bidirectional ? 2 : 1;
    const int bsize = GetRnnBiasSize(param_.num_layers, param_.state_size, direction, param_.mode);
    DType* b_ptr    = w.dptr_ + w.shape_[0] - bsize;

    DType* hy_ptr = nullptr;
    if (param_.state_outputs) {
      hy_ptr = out_data[rnn_enum::kStateOut].dptr<DType>();
    }

#if MXNET_USE_CUDNN_GE_7200
    Tensor<cpu, 1, char> host_workspace;
    int* sequence_length_cpu_int     = nullptr;
    IType* sequence_length_cpu_itype = nullptr;

    if (ctx_.dev_type == kGPU) {
      int host_workspace_bytes =
          param_.batch_size_ * sizeof(IType) + param_.batch_size_ * sizeof(int);

      host_workspace = ctx.requested[rnn_enum::kTempSpace].get_host_space_typed<1, char>(
          Shape1(host_workspace_bytes));

      sequence_length_cpu_int = reinterpret_cast<int*>(host_workspace.dptr_);
      sequence_length_cpu_itype =
          reinterpret_cast<IType*>(host_workspace.dptr_ + sizeof(int) * param_.batch_size_);

      (void)sequence_length_cpu_int;
      (void)sequence_length_cpu_itype;
    }
#endif

    if (param_.use_sequence_length) {
#if MXNET_USE_CUDNN_GE_7200
      if (ctx_.dev_type == kCPU) {
        LOG(FATAL) << "RNN use_sequence_length option is only available for cuDNN at the moment."
                   << " Not supported on CPU";
      }

      // We can assume we are on GPU for now
      size_t seq_len_input_idx = rnn_enum::kSequenceLength;
      if (param_.mode != rnn_enum::kLstm) {
        seq_len_input_idx -= 1;
      }
      IType* sequence_length_ptr_gpu = (in_data[seq_len_input_idx].get<xpu, 1, IType>(s)).dptr_;

      // Need to copy from GPU -> CPU, becuase cuDNN API requires this array on CPU memory.
      // TODO(stephenrawls): In future, allow users to pass this array on the CPU so we don't have
      //   to do this copy For now however it is required as several places in backend assume that
      //   all data arrays share the same context.
      CUDA_CALL(cudaMemcpy(sequence_length_cpu_itype,
                           sequence_length_ptr_gpu,
                           sizeof(IType) * param_.batch_size_,
                           cudaMemcpyDeviceToHost));
#else
      LOG(FATAL) << "RNN use_sequence_length option is only available for cuDNN version >= 7.2";
#endif
    }
    DType* cx_ptr = nullptr;
    DType* cy_ptr = nullptr;
    if (param_.mode == rnn_enum::kLstm) {
      cx_ptr = (in_data[rnn_enum::kStateCell].get<xpu, 3, DType>(s)).dptr_;
    }
    if (param_.mode == rnn_enum::kLstm && param_.state_outputs) {
      cy_ptr = (out_data[rnn_enum::kStateCellOut].get<xpu, 3, DType>(s)).dptr_;
    }
    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);

#if MXNET_USE_CUDNN == 1 && defined(__CUDACC__)
    if (!init_cudnn_) {
      Init(ctx, s, in_data, out_data);
    }

    // Get temp space
    int temp_size = workspace_size_;
    Tensor<gpu, 1, DType> temp_space =
        ctx.requested[rnn_enum::kTempSpace].get_space_typed<gpu, 1, DType>(
            mshadow::Shape1(temp_size), s);

#if MXNET_USE_CUDNN_GE_7200

    cudnnRNNDataLayout_t layout_t;

    if (param_.use_sequence_length) {
      // Note: Can't mempcy, sequence_length_ptr_cpu is of type Itype, not nescesarily int
      for (int i = 0; i < param_.batch_size_; ++i) {
        sequence_length_cpu_int[i] = sequence_length_cpu_itype[i];
      }
      layout_t = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
    } else {
      for (int i = 0; i < param_.batch_size_; ++i) {
        sequence_length_cpu_int[i] = param_.seq_length_;
      }
      layout_t = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
    }

    CUDNN_CALL(cudnnSetRNNDataDescriptor(x_data_desc_,
                                         dtype_,
                                         layout_t,
                                         param_.seq_length_,
                                         param_.batch_size_,
                                         param_.input_size_,
                                         sequence_length_cpu_int,
                                         reinterpret_cast<void*>(&padding_fill_)));
    int out_size =
        (param_.projection_size.has_value()) ? param_.projection_size.value() : param_.state_size;
    out_size = (param_.bidirectional) ? (out_size * 2) : out_size;
    CUDNN_CALL(cudnnSetRNNDataDescriptor(y_data_desc_,
                                         dtype_,
                                         layout_t,
                                         param_.seq_length_,
                                         param_.batch_size_,
                                         out_size,
                                         sequence_length_cpu_int,
                                         reinterpret_cast<void*>(&padding_fill_)));
    if (ctx.is_train) {
      CUDNN_CALL(cudnnSetRNNDataDescriptor(dx_data_desc_,
                                           dtype_,
                                           layout_t,
                                           param_.seq_length_,
                                           param_.batch_size_,
                                           param_.input_size_,
                                           sequence_length_cpu_int,
                                           reinterpret_cast<void*>(&padding_fill_)));
      CUDNN_CALL(cudnnSetRNNDataDescriptor(dy_data_desc_,
                                           dtype_,
                                           layout_t,
                                           param_.seq_length_,
                                           param_.batch_size_,
                                           out_size,
                                           sequence_length_cpu_int,
                                           reinterpret_cast<void*>(&padding_fill_)));
    }

    bool clip_state = param_.lstm_state_clip_min.has_value();
    bool clip_nan   = param_.lstm_state_clip_nan;
    CUDNN_CALL(cudnnRNNSetClip(s->dnn_handle_,
                               rnn_desc_,
                               clip_state ? CUDNN_RNN_CLIP_MINMAX : CUDNN_RNN_CLIP_NONE,
                               clip_nan ? CUDNN_NOT_PROPAGATE_NAN : CUDNN_PROPAGATE_NAN,
                               clip_state ? param_.lstm_state_clip_min.value() : 0.0,
                               clip_state ? param_.lstm_state_clip_max.value() : 0.0));
#endif  // MXNET_USE_CUDNN_GE_7200

    if (ctx.is_train) {
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnRNNForwardTrainingEx(s->dnn_handle_,
                                           rnn_desc_,
                                           x_data_desc_,
                                           x.dptr_,
                                           hx_desc_,
                                           hx.dptr_,
                                           cx_desc_,
                                           cx_ptr,
                                           w_desc_,
                                           w.dptr_,
                                           y_data_desc_,
                                           y.dptr_,
                                           hy_desc_,
                                           hy_ptr,
                                           cy_desc_,
                                           cy_ptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           temp_space.dptr_,
                                           workspace_byte_,
                                           reserve_space_.dptr,
                                           reserve_space_byte_));
#else
      CUDNN_CALL(cudnnRNNForwardTraining(s->dnn_handle_,
                                         rnn_desc_,
                                         param_.seq_length_,
                                         x_desc_vec_.data(),
                                         x.dptr_,
                                         hx_desc_,
                                         hx.dptr_,
                                         cx_desc_,
                                         cx_ptr,
                                         w_desc_,
                                         w.dptr_,
                                         y_desc_vec_.data(),
                                         y.dptr_,
                                         hy_desc_,
                                         hy_ptr,
                                         cy_desc_,
                                         cy_ptr,
                                         temp_space.dptr_,
                                         workspace_byte_,
                                         reserve_space_.dptr,
                                         reserve_space_byte_));
#endif  // MXNET_USE_CUDNN_GE_7200
    } else {
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnRNNForwardInferenceEx(s->dnn_handle_,
                                            rnn_desc_,
                                            x_data_desc_,
                                            x.dptr_,
                                            hx_desc_,
                                            hx.dptr_,
                                            cx_desc_,
                                            cx_ptr,
                                            w_desc_,
                                            w.dptr_,
                                            y_data_desc_,
                                            y.dptr_,
                                            hy_desc_,
                                            hy_ptr,
                                            cy_desc_,
                                            cy_ptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            temp_space.dptr_,
                                            workspace_byte_));
#else
      CUDNN_CALL(cudnnRNNForwardInference(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.seq_length_,
                                          x_desc_vec_.data(),
                                          x.dptr_,
                                          hx_desc_,
                                          hx.dptr_,
                                          cx_desc_,
                                          cx_ptr,
                                          w_desc_,
                                          w.dptr_,
                                          y_desc_vec_.data(),
                                          y.dptr_,
                                          hy_desc_,
                                          hy_ptr,
                                          cy_desc_,
                                          cy_ptr,
                                          temp_space.dptr_,
                                          workspace_byte_));
#endif  // MXNET_USE_CUDNN_GE_7200
    }
#endif  // MXNET_USE_CUDNN == 1 && defined(__CUDACC__)

#if !defined(__CUDACC__)  // cuda doesn't support C++17
    if constexpr (std::is_same<xpu, cpu>::value) {
      int projection_size = 0;
      if (param_.projection_size.has_value()) {
        projection_size = param_.projection_size.value();
      }

      // allocate temp space
      const size_t work_cpu_space_size = GetRNNWorkspaceSize(param_.seq_length_,
                                                             param_.batch_size_,
                                                             param_.state_size,
                                                             projection_size,
                                                             direction,
                                                             param_.mode);
      if (!temp_init_space_ || temp_cpu_space_size_ < work_cpu_space_size) {
        temp_cpu_space_size_ = work_cpu_space_size;
        temp_cpu_space_      = NDArray(TShape({static_cast<dim_t>(temp_cpu_space_size_)}),
                                  ctx_,
                                  false,
                                  in_data[rnn_enum::kData].type_flag_);
        temp_init_space_     = true;
      }
      DType* work_cpu_space = static_cast<DType*>(temp_cpu_space_.data().dptr_);

      if (ctx.is_train || ctx.need_grad) {
        mshadow::Random<cpu, unsigned>* prnd = ctx.requested[0].get_random<xpu, unsigned int>(s);
        std::mt19937& rnd_engine             = prnd->GetRndEngine();

        // allocate reserve space
        if (param_.projection_size.has_value()) {
          LOG(FATAL) << "No training support for LSTM with projection on CPU currently.";
        }

        const size_t r_size = GetRNNReserveSpaceSize(param_.num_layers,
                                                     direction,
                                                     param_.seq_length_,
                                                     param_.batch_size_,
                                                     param_.state_size,
                                                     param_.mode);
        if (!init_space_ || reserve_cpu_space_size_ < r_size) {
          reserve_cpu_space_size_ = r_size;
          reserve_cpu_space_      = NDArray(TShape({static_cast<dim_t>(reserve_cpu_space_size_)}),
                                       ctx_,
                                       false,
                                       in_data[rnn_enum::kData].type_flag_);
          init_space_             = true;
        }
        DType* reserve_space_ptr = static_cast<DType*>(reserve_cpu_space_.data().dptr_);

        RNNForwardTraining<DType>(work_cpu_space,
                                  reserve_space_ptr,
                                  param_.state_outputs,
                                  param_.num_layers,
                                  direction,
                                  param_.seq_length_,
                                  param_.batch_size_,
                                  param_.input_size_,
                                  param_.state_size,
                                  x.dptr_,
                                  hx.dptr_,
                                  cx_ptr,
                                  w.dptr_,
                                  b_ptr,
                                  y.dptr_,
                                  hy_ptr,
                                  cy_ptr,
                                  param_.p,
                                  param_.mode,
                                  rnd_engine);
      } else {
        RNNForwardInference<DType>(work_cpu_space,
                                   param_.state_outputs,
                                   param_.num_layers,
                                   direction,
                                   param_.seq_length_,
                                   param_.batch_size_,
                                   param_.input_size_,
                                   param_.state_size,
                                   projection_size,
                                   x.dptr_,
                                   hx.dptr_,
                                   cx_ptr,
                                   w.dptr_,
                                   b_ptr,
                                   y.dptr_,
                                   hy_ptr,
                                   cy_ptr,
                                   param_.mode);
      }
    }
#endif
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<TBlob>& out_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";

    size_t num_inputs = GetRnnNumInputs(param_);

    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    }

    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);
    CHECK_EQ(in_grad.size(), num_inputs);
    CHECK_EQ(out_grad.size(), num_outputs);
    CHECK_EQ(req.size(), num_inputs);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo) << "AddTo is not supported for state";
    Stream<xpu>* s = ctx.get_stream<xpu>();
    // get input + output tensors
    Tensor<xpu, 3, DType> x   = in_data[rnn_enum::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> dx  = in_grad[rnn_enum::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 1, DType> w   = in_data[rnn_enum::kParams].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> dw  = in_grad[rnn_enum::kParams].get<xpu, 1, DType>(s);
    Tensor<xpu, 3, DType> hx  = in_data[rnn_enum::kState].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> dhx = in_grad[rnn_enum::kState].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> y   = out_data[rnn_enum::kOut].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> dy  = out_grad[rnn_enum::kOut].get<xpu, 3, DType>(s);

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(dw.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(dhx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);
    CHECK_EQ(dy.CheckContiguous(), true);
    CHECK_EQ(dx.CheckContiguous(), true);

    if (req[rnn_enum::kParams] != kAddTo) {
      dw = mshadow::expr::ScalarExp<DType>(0.0f);
    }

    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    const int direction = param_.bidirectional ? 2 : 1;
    const int bsize = GetRnnBiasSize(param_.num_layers, param_.state_size, direction, param_.mode);

    DType* db_ptr = dw.dptr_ + w.shape_[0] - bsize;

    DType* dhy_ptr = nullptr;
    if (param_.state_outputs) {
      dhy_ptr = out_grad[rnn_enum::kStateOut].dptr<DType>();
    }

    DType* dcx_ptr = nullptr;
    DType* dcy_ptr = nullptr;
    DType* cx_ptr  = nullptr;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo) << "AddTo is not supported for state cell";
      cx_ptr  = (in_data[rnn_enum::kStateCell].get<xpu, 3, DType>(s)).dptr_;
      dcx_ptr = (in_grad[rnn_enum::kStateCell].get<xpu, 3, DType>(s)).dptr_;
    }
    if ((param_.mode == rnn_enum::kLstm) && param_.state_outputs) {
      dcy_ptr = (out_grad[rnn_enum::kStateCellOut].get<xpu, 3, DType>(s)).dptr_;
    }

#if MXNET_USE_CUDNN == 1 && defined(__CUDACC__)
    if (!init_cudnn_) {
      Init(ctx, s, in_data, out_data);
    }

    // Get temp space
    int temp_size = workspace_size_;
    Tensor<gpu, 1, DType> temp_space =
        ctx.requested[rnn_enum::kTempSpace].get_space_typed<gpu, 1, DType>(
            mshadow::Shape1(temp_size), s);

#if MXNET_USE_CUDNN_GE_7200
    CUDNN_CALL(cudnnRNNBackwardDataEx(s->dnn_handle_,
                                      rnn_desc_,
                                      y_data_desc_,
                                      y.dptr_,
                                      dy_data_desc_,
                                      dy.dptr_,
                                      nullptr,
                                      nullptr,
                                      dhy_desc_,
                                      dhy_ptr,
                                      dcy_desc_,
                                      dcy_ptr,
                                      w_desc_,
                                      w.dptr_,
                                      hx_desc_,
                                      hx.dptr_,
                                      cx_desc_,
                                      cx_ptr,
                                      dx_data_desc_,
                                      dx.dptr_,
                                      dhx_desc_,
                                      dhx.dptr_,
                                      dcx_desc_,
                                      dcx_ptr,
                                      nullptr,
                                      nullptr,
                                      temp_space.dptr_,
                                      workspace_byte_,
                                      reserve_space_.dptr,
                                      reserve_space_byte_));
    SyncDgrad();
    if (req[rnn_enum::kParams] != kNullOp) {
      CUDNN_CALL(cudnnRNNBackwardWeightsEx(s->dnn_handle_,
                                           rnn_desc_,
                                           x_data_desc_,
                                           x.dptr_,
                                           hx_desc_,
                                           hx.dptr_,
                                           y_data_desc_,
                                           y.dptr_,
                                           temp_space.dptr_,
                                           workspace_byte_,
                                           dw_desc_,
                                           dw.dptr_,
                                           reserve_space_.dptr,
                                           reserve_space_byte_));
    }
#else
    CUDNN_CALL(cudnnRNNBackwardData(s->dnn_handle_,
                                    rnn_desc_,
                                    param_.seq_length_,
                                    y_desc_vec_.data(),
                                    y.dptr_,
                                    dy_desc_vec_.data(),
                                    dy.dptr_,
                                    dhy_desc_,
                                    dhy_ptr,
                                    dcy_desc_,
                                    dcy_ptr,
                                    w_desc_,
                                    w.dptr_,
                                    hx_desc_,
                                    hx.dptr_,
                                    cx_desc_,
                                    cx_ptr,
                                    dx_desc_vec_.data(),
                                    dx.dptr_,
                                    dhx_desc_,
                                    dhx.dptr_,
                                    dcx_desc_,
                                    dcx_ptr,
                                    temp_space.dptr_,
                                    workspace_byte_,
                                    reserve_space_.dptr,
                                    reserve_space_byte_));
    SyncDgrad();
    if (req[rnn_enum::kParams] != kNullOp) {
      CUDNN_CALL(cudnnRNNBackwardWeights(s->dnn_handle_,
                                         rnn_desc_,
                                         param_.seq_length_,
                                         x_desc_vec_.data(),
                                         x.dptr_,
                                         hx_desc_,
                                         hx.dptr_,
                                         y_desc_vec_.data(),
                                         y.dptr_,
                                         temp_space.dptr_,
                                         workspace_byte_,
                                         dw_desc_,
                                         dw.dptr_,
                                         reserve_space_.dptr,
                                         reserve_space_byte_));
    }
#endif  // MXNET_USE_CUDNN_GE_7200
#endif  // MXNET_USE_CUDNN == 1 && defined(__CUDACC__)

    if (ctx_.dev_type == kCPU) {
      int projection_size = 0;
      if (param_.projection_size.has_value()) {
        // TODO(zixuanweeei): Add training support for LSTM with projection on CPU.
        // projection_size = param_.projection_size.value();
        LOG(FATAL) << "No training support for LSTM with projection on CPU currently.";
      }

      // allocate temp space
      const size_t work_cpu_space_size = GetRNNWorkspaceSize(param_.seq_length_,
                                                             param_.batch_size_,
                                                             param_.state_size,
                                                             projection_size,
                                                             direction,
                                                             param_.mode);
      if (!temp_init_space_ || temp_cpu_space_size_ != work_cpu_space_size) {
        LOG(FATAL) << "Check temp init error";
      }
      DType* work_cpu_space = static_cast<DType*>(temp_cpu_space_.data().dptr_);
      size_t r_size         = GetRNNReserveSpaceSize(param_.num_layers,
                                             direction,
                                             param_.seq_length_,
                                             param_.batch_size_,
                                             param_.state_size,
                                             param_.mode);

      if (!init_space_ || reserve_cpu_space_size_ != r_size) {
        LOG(FATAL) << "Check forward init error";
      }

      DType* reserve_space_ptr = static_cast<DType*>(reserve_cpu_space_.data().dptr_);
      RNNBackward<DType>(work_cpu_space,
                         reserve_space_ptr,
                         param_.num_layers,
                         direction,
                         param_.seq_length_,
                         param_.batch_size_,
                         param_.input_size_,
                         param_.state_size,
                         x.dptr_,
                         hx.dptr_,
                         cx_ptr,
                         w.dptr_,
                         y.dptr_,
                         dy.dptr_,
                         dhy_ptr,
                         dcy_ptr,
                         dx.dptr_,
                         dhx.dptr_,
                         dcx_ptr,
                         dw.dptr_,
                         db_ptr,
                         req[rnn_enum::kData],
                         req[rnn_enum::kParams],
                         req[rnn_enum::kState],
                         // State cell should be present for LSTMs, but is absent for other RNNs.
                         param_.mode == rnn_enum::kLstm ? req[rnn_enum::kStateCell] : kNullOp,
                         param_.p,
                         param_.mode);
    }
  }

 private:
  inline void Init(const OpContext& ctx,
                   mshadow::Stream<xpu>* s,
                   const std::vector<TBlob>& in_data,
                   const std::vector<TBlob>& out_data) {
    using namespace mshadow;

    size_t num_inputs = GetRnnNumInputs(param_);
    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3U : 2U;
    }

    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);

#if MXNET_USE_CUDNN == 1 && defined(__CUDACC__)
    format_ = CUDNN_TENSOR_NCHW;

    if (!init_cudnn_) {
      init_cudnn_ = true;
      // get input + output tensors
      Tensor<xpu, 3, DType> x = in_data[rnn_enum::kData].get<xpu, 3, DType>(s);
      Tensor<xpu, 1, DType> w = in_data[rnn_enum::kParams].get<xpu, 1, DType>(s);
      param_.seq_length_      = x.shape_[0];
      param_.batch_size_      = x.shape_[1];
      param_.input_size_      = x.shape_[2];

      // Tensor Descriptors
      std::vector<cudnnTensorDescriptor_t> x_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> y_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> dx_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> dy_vec(param_.seq_length_);
      int dimA[3];
      int strideA[3];
      for (int i = 0; i < param_.seq_length_; i++) {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_vec[i]));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&y_vec[i]));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_vec[i]));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_vec[i]));

        dimA[0]    = param_.batch_size_;
        dimA[1]    = param_.input_size_;
        dimA[2]    = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(cudnnSetTensorNdDescriptor(x_vec[i], dtype_, 3, dimA, strideA));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(dx_vec[i], dtype_, 3, dimA, strideA));
        dimA[0]    = param_.batch_size_;
        dimA[1]    = param_.bidirectional ? param_.state_size * 2 : param_.state_size;
        dimA[2]    = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(cudnnSetTensorNdDescriptor(y_vec[i], dtype_, 3, dimA, strideA));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(dy_vec[i], dtype_, 3, dimA, strideA));
      }
      x_desc_vec_  = x_vec;
      y_desc_vec_  = y_vec;
      dx_desc_vec_ = dx_vec;
      dy_desc_vec_ = dy_vec;

      // set the state tensors
      dimA[0]    = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimA[1]    = param_.batch_size_;
      dimA[2]    = param_.state_size;
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;
#if MXNET_USE_CUDNN_GE_7200
      int dimB[3];
      int strideB[3];
      dimB[0] = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimB[1] = param_.batch_size_;
      dimB[2] =
          param_.projection_size.has_value() ? param_.projection_size.value() : param_.state_size;
      strideB[0] = dimB[2] * dimB[1];
      strideB[1] = dimB[2];
      strideB[2] = 1;
#endif  // MXNET_USE_CUDNN_GE_7200
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hx_desc_, dtype_, 3, dimB, strideB));
#else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hx_desc_, dtype_, 3, dimA, strideA));
#endif  // MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(cx_desc_, dtype_, 3, dimA, strideA));
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hy_desc_, dtype_, 3, dimB, strideB));
#else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hy_desc_, dtype_, 3, dimA, strideA));
#endif  // MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(cy_desc_, dtype_, 3, dimA, strideA));
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhx_desc_, dtype_, 3, dimB, strideB));
#else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhx_desc_, dtype_, 3, dimA, strideA));
#endif  // MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dcx_desc_, dtype_, 3, dimA, strideA));
#if MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhy_desc_, dtype_, 3, dimB, strideB));
#else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhy_desc_, dtype_, 3, dimA, strideA));
#endif  // MXNET_USE_CUDNN_GE_7200
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dcy_desc_, dtype_, 3, dimA, strideA));

      // Create Dropout descriptors
      ctx.requested[rnn_enum::kCuDNNDropoutDescSpace].get_cudnn_dropout_desc(
          &dropout_desc_, s, param_.p);

      // RNN descriptors
      // adopt pseudo-fp16 for all architectures
      cudnnDataType_t dtype_with_fallback_ =
          (cudnnGetVersion() >= 7500 && dtype_ == CUDNN_DATA_HALF) ? CUDNN_DATA_FLOAT : dtype_;
      cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;
      dgrad_sync_needed_      = (rnn_algo == CUDNN_RNN_ALGO_STANDARD) && param_.bidirectional;
      CUDNN_CALL(cudnnSetRNNDescriptor_v6(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.state_size,
                                          param_.num_layers,
                                          dropout_desc_,
                                          input_mode_,
                                          direction_,
                                          mode_,
                                          rnn_algo,
                                          dtype_with_fallback_));
      cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
      if (cudnn_tensor_core_ && rnn_algo == CUDNN_RNN_ALGO_STANDARD) {
        math_type = CUDNN_TENSOR_OP_MATH;
      }
#if CUDNN_VERSION >= 7200
      if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
          (DataType<DType>::kFlag != kFloat16)) {
        math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
      }
#endif
      CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_, math_type));
#if MXNET_USE_CUDNN_GE_7200
      if (param_.projection_size.has_value()) {
        CUDNN_CALL(cudnnSetRNNProjectionLayers(
            s->dnn_handle_, rnn_desc_, param_.projection_size.value(), 0));
      }
      if (param_.use_sequence_length) {
        CUDNN_CALL(cudnnSetRNNPaddingMode(rnn_desc_, CUDNN_RNN_PADDED_IO_ENABLED));
      }
#endif  // MXNET_USE_CUDNN_GE_7200

      // Get temp space sizes
      CUDNN_CALL(cudnnGetRNNWorkspaceSize(
          s->dnn_handle_, rnn_desc_, param_.seq_length_, x_desc_vec_.data(), &workspace_byte_));
      CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
          s->dnn_handle_, rnn_desc_, param_.seq_length_, x_desc_vec_.data(), &reserve_space_byte_));
      workspace_size_ = workspace_byte_ / sizeof(DType);
      // Allocate the reserve space
      reserve_space_ = Storage::Get()->Alloc(reserve_space_byte_, Context::GPU(s->dev_id));
      reserve_space_.profiler_scope = "cudnn_rnn:";
      reserve_space_.name           = "reserve_space";
      profiler::GpuDeviceStorageProfiler::Get()->UpdateStorageInfo(reserve_space_);
      // Check that number of params are correct
      size_t cudnn_param_size;
      CUDNN_CALL(cudnnGetRNNParamsSize(
          s->dnn_handle_, rnn_desc_, x_desc_vec_[0], &cudnn_param_size, dtype_));
      CHECK_EQ(w.shape_[0] * sizeof(DType), cudnn_param_size);
      // Set param descriptors
      int dim_w[3] = {1, 1, 1};
      dim_w[0]     = w.shape_[0];
      CUDNN_CALL(cudnnSetFilterNdDescriptor(w_desc_, dtype_, format_, 3, dim_w));
      CUDNN_CALL(cudnnSetFilterNdDescriptor(dw_desc_, dtype_, format_, 3, dim_w));

      // Query weight layout
      // cudnnFilterDescriptor_t m_desc;
      // CHECK_EQ(cudnnCreateFilterDescriptor(&m_desc), CUDNN_STATUS_SUCCESS);
      // DType *p;
      // int n = 2;
      // int64_t last = 0;
      // if (param_.mode == rnn_enum::kLstm) n = 8;
      // else if (param_.mode == rnn_enum::kGru) n = 6;

      // for (int i = 0; i < param_.num_layers*(param_.bidirectional?2:1); ++i) {
      //   for (int j = 0; j < n; ++j) {
      //     CHECK_EQ(cudnnGetRNNLinLayerMatrixParams(s->dnn_handle_, rnn_desc_,
      //       i, x_desc_vec_[0], w_desc_, 0, j, m_desc, (void**)&p), CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << ((int64_t)(p - nullptr))/sizeof(DType) - last;
      //     last = ((int64_t)(p - nullptr))/sizeof(DType);
      //     cudnnDataType_t t;
      //     cudnnTensorFormat_t f;
      //     int ndim = 5;
      //     int dims[5] = {0, 0, 0, 0, 0};
      //     CHECK_EQ(cudnnGetFilterNdDescriptor(m_desc, ndim, &t, &f, &ndim, &dims[0]),
      //       CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << "w: " <<  i << " " << j << " " << ((int64_t)(p - nullptr))/sizeof(DType);
      //     for (int i = 0; i < ndim; ++i) LOG(INFO) << dims[i];
      //   }
      // }

      // for (int i = 0; i < param_.num_layers*(param_.bidirectional?2:1); ++i) {
      //   for (int j = 0; j < n; ++j) {
      //     CHECK_EQ(cudnnGetRNNLinLayerBiasParams(s->dnn_handle_, rnn_desc_, i, x_desc_vec_[0],
      //       w_desc_, 0, j, m_desc, (void**)&p), CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << ((int64_t)(p - nullptr))/sizeof(DType) - last;
      //     last = ((int64_t)(p - nullptr))/sizeof(DType);
      //     LOG(INFO) << "b: " << i << " " << j << " " << ((int64_t)(p - nullptr))/sizeof(DType);
      //   }
      // }
    }
#endif  // MXNET_USE_CUDNN == 1 && defined(__CUDACC__)
  }
  // naive private variables used in CPU Context
  bool init_space_, temp_init_space_;
  size_t reserve_cpu_space_size_, temp_cpu_space_size_;
  NDArray reserve_cpu_space_, temp_cpu_space_;

#if MXNET_USE_CUDNN == 1 && defined(__CUDACC__)
  // cuDNN versions up to and including v7.6.4 did not sync a last dgrad kernel back to the main
  // cudnn handle's stream (non-persistant algo, bidirectional only).  This could result in silent
  // non-determinstic failures with very low probability, seen more often when wgrad is bypassed.
  inline void SyncDgrad() {
    if (CUDNN_VERSION <= 7604 && dgrad_sync_needed_) {
      // Without blocking the CPU, create a synchronization point of all current GPU activity.  No
      // need to call cudaStreamWaitEvent- cudaEventRecord on the legacy default stream suffices.
      if (!dgrad_sync_event_created_) {
        CUDA_CALL(cudaEventCreateWithFlags(&dgrad_sync_event_, cudaEventDisableTiming));
        dgrad_sync_event_created_ = true;
      }
      CUDA_CALL(cudaEventRecord(dgrad_sync_event_, cudaStreamLegacy));
    }
  }
#endif  // MXNET_USE_CUDNN == 1 && defined(__CUDACC__)

#if MXNET_USE_CUDNN == 1
  cudnnDataType_t dtype_;
  bool init_cudnn_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnRNNMode_t mode_;
  cudnnDirectionMode_t direction_;
  cudnnRNNInputMode_t input_mode_;
  cudnnDropoutDescriptor_t dropout_desc_;
  Storage::Handle reserve_space_;
  size_t workspace_byte_, reserve_space_byte_;
  int workspace_size_;
  std::vector<cudnnTensorDescriptor_t> x_desc_vec_, y_desc_vec_, dx_desc_vec_, dy_desc_vec_;
#if MXNET_USE_CUDNN_GE_7200
  cudnnRNNDataDescriptor_t x_data_desc_, y_data_desc_, dx_data_desc_, dy_data_desc_;
  DType padding_fill_ = 0;
#endif  // MXNET_USE_CUDNN_GE_7200
  cudnnTensorDescriptor_t hx_desc_, cx_desc_;
  cudnnTensorDescriptor_t hy_desc_, cy_desc_;
  cudnnTensorDescriptor_t dhx_desc_, dcx_desc_;
  cudnnTensorDescriptor_t dhy_desc_, dcy_desc_;

  cudnnFilterDescriptor_t w_desc_, dw_desc_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;

  cudnnTensorFormat_t format_;
  cudaEvent_t dgrad_sync_event_;
  bool dgrad_sync_event_created_ = false;
  bool dgrad_sync_needed_        = false;
#endif  // MXNET_USE_CUDNN
};      //  class RNNOp

template <typename xpu>
void RNNStatefulCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  int dtype = inputs[rnn_enum::kData].type_flag_;

  // Hacky. This relies on fact that seq-len type is either the last input,
  // or we aren't using seq-len input and this type should be same as dtype.
  // Would prefer direct access to RNNParam object here but not sure how to get.
  int itype = inputs[inputs.size() - 1].type_flag_;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(itype, IType, {
      RNNOp<xpu, DType, IType>& op = state.get_state<RNNOp<xpu, DType, IType>>();
      op.Forward(ctx, inputs, req, outputs);
    });
  });
}

/*
index description
0: x
1: w
2: hx
3: y
4: dy
5: hy
6: dhy
7: cx
8: cy
9: dcy
*/
template <typename xpu>
void RNNStatefulGradCompute(const OpStatePtr& state,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> out_data{inputs[3]};
  std::vector<TBlob> out_grad{inputs[4]};
  const std::vector<TBlob>& in_grad = outputs;

  int dtype = inputs[rnn_enum::kData].type_flag_;

  // Hacky. This relies on fact that seq-len type is either the last input,
  // or we aren't using seq-len input and this type should be same as dtype.
  // Would prefer direct access to RNNParam object here but not sure how to get.
  int itype = outputs[outputs.size() - 1].type_flag_;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(itype, IType, {
      RNNOp<xpu, DType, IType>& op = state.get_state<RNNOp<xpu, DType, IType>>();
      const RNNParam& param        = op.param_;
      int index                    = 5;
      if (param.state_outputs) {
        out_data.push_back(inputs[index++]);
        out_grad.push_back(inputs[index++]);
      }

      if (param.mode == rnn_enum::kLstm) {
        in_data.push_back(inputs[index++]);
        if (param.state_outputs) {
          out_data.push_back(inputs[index++]);
          out_grad.push_back(inputs[index]);
        }
      }

      if (param.use_sequence_length) {
        size_t seq_len_input_idx = rnn_enum::kSequenceLength;
        if (param.mode != rnn_enum::kLstm) {
          seq_len_input_idx -= 1;
        }
        in_data.push_back(outputs[seq_len_input_idx]);
      }

      op.Backward(ctx, out_grad, in_data, out_data, req, in_grad);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RNN_INL_H_
