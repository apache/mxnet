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
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include "./rnn_impl.h"

namespace mxnet {
namespace op {

namespace rnn_enum {
  enum RNNOpInputs {kData, kParams, kState, kStateCell};
  enum RNNOpOutputs {kOut, kStateOut, kStateCellOut};
  enum RNNModeType {kRnnRelu, kRnnTanh, kLstm, kGru};
  enum RNNOpResource {kTempSpace};
}

inline int GetRnnParamSize(int num_layer,
                           int input_size,
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
  int size1 = (input_size + state_size + 2) * size;  // first layer size
  int size2 = (state_size * direction + state_size + 2) * size;  // other layers size
  if (projection_size.has_value()) {
    int proj_size = projection_size.value();
    size1 = (input_size + proj_size + 2) * size;
    size2 = (proj_size * direction + proj_size + 2) * size;
  }
  int param_size = size1 + (num_layer - 1) * size2;
  if (projection_size.has_value()) {
    param_size += projection_size.value() * state_size * num_layer * direction;
  }
  return param_size;
}

inline int GetRnnBiasSize(int num_layer,
                           int state_size,
                           int direction,
                           int mode) {
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

inline size_t GetRNNWorkspaceSize(int seq_length,
                                  int batch_size,
                                  int hidden_size,
                                  int direction,
                                  int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = (seq_length + 1) * batch_size * hidden_size * 4 + batch_size * hidden_size * 2
             + seq_length * batch_size * hidden_size * direction + hidden_size * seq_length * 8;
      break;
    case rnn_enum::kGru:
      size = seq_length * batch_size * hidden_size * direction * 4 + batch_size * hidden_size * 8;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      size = seq_length * batch_size * hidden_size * direction * 2 + batch_size * hidden_size * 4;
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

inline size_t GetRNNReserveSpaceSize(int num_layer,
                                     int direction,
                                     int seq_length,
                                     int batch_size,
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

struct RNNParam : public dmlc::Parameter<RNNParam> {
  uint32_t state_size;
  uint32_t num_layers;
  bool bidirectional, state_outputs;
  int mode;
  float p;
  int seq_length_, batch_size_, input_size_;
  dmlc::optional<int> projection_size;
  dmlc::optional<double> lstm_state_clip_min, lstm_state_clip_max;
  bool lstm_state_clip_nan;

  DMLC_DECLARE_PARAMETER(RNNParam) {
    DMLC_DECLARE_FIELD(state_size)
    .describe("size of the state for each layer");

    DMLC_DECLARE_FIELD(num_layers)
    .describe("number of stacked layers");

    DMLC_DECLARE_FIELD(bidirectional).set_default(false)
    .describe("whether to use bidirectional recurrent layers");

    DMLC_DECLARE_FIELD(mode)
    .add_enum("rnn_relu", rnn_enum::kRnnRelu)
    .add_enum("rnn_tanh", rnn_enum::kRnnTanh)
    .add_enum("lstm", rnn_enum::kLstm)
    .add_enum("gru", rnn_enum::kGru)
    .describe("the type of RNN to compute");

    DMLC_DECLARE_FIELD(p).set_default(0.)
    .set_range(0, 1)
    .describe("drop rate of the dropout on the outputs of each RNN layer, except the last layer.");

    DMLC_DECLARE_FIELD(state_outputs).set_default(false)
    .describe("Whether to have the states as symbol outputs.");

    DMLC_DECLARE_FIELD(projection_size)
    .set_default(dmlc::optional<int>())
    .describe("size of project size");

    DMLC_DECLARE_FIELD(lstm_state_clip_min)
    .set_default(dmlc::optional<double>())
    .describe("Minimum clip value of LSTM states. This option must be used together with "
              "lstm_state_clip_max.");

    DMLC_DECLARE_FIELD(lstm_state_clip_max)
    .set_default(dmlc::optional<double>())
    .describe("Maximum clip value of LSTM states. This option must be used together with "
              "lstm_state_clip_min.");

    DMLC_DECLARE_FIELD(lstm_state_clip_nan)
    .set_default(false)
    .describe("Whether to stop NaN from propagating in state by clipping it to min/max. "
              "If clipping range is not specified, this option is ignored.");
  }
};

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
                        const int seq_length,
                        const int batch_size,
                        const int input_size,
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
                        int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      LstmForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                 batch_size, input_size, state_size, x_ptr, hx_ptr, cx_ptr,
                                 w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, dropout);
      break;
    case rnn_enum::kGru:
      GruForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                batch_size, input_size, state_size, x_ptr, hx_ptr,
                                w_ptr, y_ptr, hy_ptr, dropout);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                       batch_size, input_size, state_size, x_ptr, hx_ptr,
                                       w_ptr, y_ptr, hy_ptr, dropout, mode);
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
                         const int seq_length,
                         const int batch_size,
                         const int input_size,
                         const int state_size,
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
      LstmForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                  batch_size, input_size, state_size, x_ptr, hx_ptr, cx_ptr,
                                  w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr);
      break;
    case rnn_enum::kGru:
      GruForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                 batch_size, input_size, state_size, x_ptr, hx_ptr,
                                 w_ptr, y_ptr, hy_ptr);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                        batch_size, input_size, state_size, x_ptr, hx_ptr,
                                        w_ptr, y_ptr, hy_ptr, mode);
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
                 const int seq_length,
                 const int batch_size,
                 const int input_size,
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
      LstmBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                          input_size, state_size, x_ptr, hx_ptr, cx_ptr, w_ptr, y_ptr,
                          dy_ptr, dhy_ptr, dcy_ptr, dx_ptr, dhx_ptr, dcx_ptr, dw_ptr, db_ptr,
                          req_data, req_params, req_state, req_statecell, dropout);
      break;
    case rnn_enum::kGru:
      GruBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                         input_size, state_size, x_ptr, hx_ptr, w_ptr,
                         dy_ptr, dhy_ptr, dx_ptr, dhx_ptr, dw_ptr,
                         req_data, req_params, req_state, dropout);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                                input_size, state_size, x_ptr, hx_ptr, w_ptr,
                                dy_ptr, dhy_ptr, dx_ptr, dhx_ptr, dw_ptr,
                                req_data, req_params, req_state, dropout, mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template<typename DType>
class RNNOp {
 public:
  explicit RNNOp(RNNParam p)
    :param_(p), init_space_(false), reserve_space_size_(0) {
    if (param_.projection_size.has_value()) {
      LOG(FATAL) << "hidden layer projection is only supported for GPU with CuDNN later than 7.1.1";
    }
    if (param_.lstm_state_clip_min.has_value()
        || param_.lstm_state_clip_max.has_value()) {
      LOG(FATAL) << "LSTM state clipping is only supported for GPU with CuDNN later than 7.2.1";
    }
  }

  ~RNNOp() {
    if (init_space_) {
      Storage::Get()->Free(reserve_space_);
      init_space_ = false;
    }
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";

    size_t num_inputs = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    }
    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensor
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
    Tensor<cpu, 1, DType> w = in_data[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> hx = in_data[rnn_enum::kState].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> y = out_data[rnn_enum::kOut].get<cpu, 3, DType>(s);
    CHECK(x.CheckContiguous());
    CHECK(w.CheckContiguous());
    CHECK(hx.CheckContiguous());
    CHECK(y.CheckContiguous());
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];
    const int direction = param_.bidirectional ? 2 : 1;
    const int bsize = GetRnnBiasSize(param_.num_layers, param_.state_size, direction, param_.mode);
    DType* b_ptr = w.dptr_ + w.shape_[0] - bsize;

    DType* hy_ptr = NULL;
    if (param_.state_outputs) {
      hy_ptr = out_data[rnn_enum::kStateOut].dptr<DType>();
    }
    DType* cx_ptr = NULL;
    DType* cy_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      cx_ptr = in_data[rnn_enum::kStateCell].dptr<DType>();
      if (param_.state_outputs) {
        cy_ptr = out_data[rnn_enum::kStateCellOut].dptr<DType>();
      }
    }

    // allocate temp space
    const size_t workspace_size = GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                                                      param_.state_size, direction, param_.mode);
    Tensor<cpu, 1, DType> workspace = ctx.requested[rnn_enum::kTempSpace]
        .get_space_typed<cpu, 1, DType>(Shape1(workspace_size), s);
    if (ctx.is_train) {
      const size_t r_size = GetRNNReserveSpaceSize(param_.num_layers, direction,
                                                   param_.seq_length_, param_.batch_size_,
                                                   param_.state_size, param_.mode);
      if (init_space_ && reserve_space_size_ < r_size) {
        Storage::Get()->Free(reserve_space_);
        init_space_ = false;
      }
      if (!init_space_) {
        reserve_space_ = Storage::Get()->Alloc(r_size * sizeof(DType), Context::CPU());
        reserve_space_size_ = r_size;
        init_space_ = true;
      }

      DType* reserve_space_ptr = static_cast<DType*>(reserve_space_.dptr);

      RNNForwardTraining<DType>(workspace.dptr_,
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
                                param_.mode);
    } else {
      RNNForwardInference<DType>(workspace.dptr_,
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
                                 param_.mode);
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";

    size_t num_inputs = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
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
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensors
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
    Tensor<cpu, 1, DType> w = in_data[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> hx = in_data[rnn_enum::kState].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> y = out_data[rnn_enum::kOut].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> dx = in_grad[rnn_enum::kData].get<cpu, 3, DType>(s);
    Tensor<cpu, 1, DType> dw = in_grad[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> dhx = in_grad[rnn_enum::kState].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> dy = out_grad[rnn_enum::kOut].get<cpu, 3, DType>(s);
    CHECK(x.CheckContiguous());
    CHECK(w.CheckContiguous());
    CHECK(hx.CheckContiguous());
    CHECK(y.CheckContiguous());
    CHECK(dx.CheckContiguous());
    CHECK(dw.CheckContiguous());
    CHECK(dhx.CheckContiguous());
    CHECK(dy.CheckContiguous());
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    const int direction = param_.bidirectional ? 2 : 1;
    const int bsize = GetRnnBiasSize(param_.num_layers, param_.state_size, direction, param_.mode);

    DType* db_ptr = dw.dptr_ + w.shape_[0] - bsize;

    DType * dhy_ptr = NULL;
    if (param_.state_outputs) {
      dhy_ptr = out_grad[rnn_enum::kStateOut].dptr<DType>();
    }

    DType * cx_ptr = NULL;
    DType * dcx_ptr = NULL;
    DType * dcy_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo) << "AddTo is not supported for state cell";
      cx_ptr = in_data[rnn_enum::kStateCell].dptr<DType>();
      dcx_ptr = in_grad[rnn_enum::kStateCell].dptr<DType>();
      if (param_.state_outputs) {
        dcy_ptr = out_grad[rnn_enum::kStateCellOut].dptr<DType>();
      }
    }

    // allocate temp space
    const size_t workspace_size = GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                                                      param_.state_size, direction, param_.mode);
    Tensor<cpu, 1, DType> workspace = ctx.requested[rnn_enum::kTempSpace]
        .get_space_typed<cpu, 1, DType>(Shape1(workspace_size), s);

    size_t r_size = GetRNNReserveSpaceSize(param_.num_layers, direction,
                                           param_.seq_length_, param_.batch_size_,
                                           param_.state_size, param_.mode);

    if (!init_space_ || reserve_space_size_ != r_size) {
      LOG(FATAL) << "Check forward init error";
    }

    DType* reserve_space_ptr = static_cast<DType*>(reserve_space_.dptr);
    RNNBackward<DType>(workspace.dptr_,
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

  RNNParam param_;

 private:
  bool init_space_;
  size_t reserve_space_size_;
  Storage::Handle reserve_space_;
};  // class RNNOp

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RNN_INL_H_
