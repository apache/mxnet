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
 * \author Sebastian Bodenstein, Shu Zhang(shu.zhang@intel.com)
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
#include "./rnn_impl.hpp"

namespace mxnet {
namespace op {

namespace rnn_enum {
  enum RNNOpInputs {kData, kParams, kState, kStateCell};
  enum RNNOpOutputs {kOut, kStateOut, kStateCellOut};
  enum RNNModeType {kRnnRelu, kRnnTanh, kLstm, kGru};
  enum RNNOpResource {kTempSpace};
}

// A utility function to calculate input size
inline int rnn_single_param_size(int inputSize,
                                 int hiddenSize,
                                 int mode) {
  int size = hiddenSize * (hiddenSize + inputSize + 2);
  // Different RNN's have different num weights
  switch (mode) {
    case rnn_enum::kRnnRelu:
      size *= 1;
      break;
    case rnn_enum::kRnnTanh:
      size *= 1;
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

inline int rnn_param_size(int layerNum,
                          int inputSize,
                          int hiddenSize,
                          bool bidirectional,
                          int mode) {
  // get size of first layer
  int size = rnn_single_param_size(inputSize, hiddenSize, mode);
  // get size of remaining layers
  if (bidirectional) {
    size += (layerNum - 1) * rnn_single_param_size(2 * hiddenSize, hiddenSize, mode);
    size *= 2;
  } else {
    size += (layerNum - 1) * rnn_single_param_size(hiddenSize, hiddenSize, mode);
  }
  return size;
}

inline size_t GetRNNWorkspaceSize(int seq_length,
                                  int batch_size,
                                  int hidden_size,
                                  int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kRnnRelu:
      break;
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      size = (seq_length + 1) * batch_size * hidden_size * 4 + batch_size * hidden_size;
      break;
    case rnn_enum::kGru:
      break;
  }
  return size;
}

inline size_t GetRNNReserveSpaceSize(int seq_length,
                                     int batch_size,
                                     int hidden_size,
                                     int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kRnnRelu:
      break;
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      size = seq_length * batch_size * hidden_size * 6;
      break;
    case rnn_enum::kGru:
      break;
  }
  return size;
}

struct RNNParam : public dmlc::Parameter<RNNParam> {
  uint32_t state_size;
  uint32_t num_layers;
  bool bidirectional, state_outputs;
  int mode;
  float p, pkeep_;
  int seq_length_, batch_size_, input_size_;
  bool lstm_q_;  // whether type is lstm

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
    .describe("Dropout probability, fraction of the input that gets dropped out at training time");

    DMLC_DECLARE_FIELD(state_outputs).set_default(false)
    .describe("Whether to have the states as symbol outputs.");
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
 *          w_ptr: Pointer of tensor w containing weights and bias.
 *          y_ptr: Pointer of tensor y containing the features of the output features from the
 *                 last layers of the RNN. y's shape is [seq_length, batch_size, state_size]
 *          hy_ptr: Pointer of tensor hy containing the hidden state for t=seq_length.
 *                  hy's shape is [num_layers, batch_size, state_size]
 *          cy_ptr: Only used in lstm mode. pointer of tensor cy  containing the cell state
 *                  for t=seq_length. cy' shape is [num_layers, batch_size, state_size]
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
                        DType* y_ptr,
                        DType* hy_ptr,
                        DType* cy_ptr,
                        int mode) {
  switch (mode) {
    case rnn_enum::kRnnRelu:
      break;
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      LstmForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                 batch_size, input_size, state_size, x_ptr, hx_ptr, cx_ptr,
                                 w_ptr, y_ptr, hy_ptr, cy_ptr);
      break;
    case rnn_enum::kGru:
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
                         DType* y_ptr,
                         DType* hy_ptr,
                         DType* cy_ptr,
                         int mode) {
  switch (mode) {
    case rnn_enum::kRnnRelu:
      break;
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      LstmForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                  batch_size, input_size, state_size, x_ptr, hx_ptr, cx_ptr,
                                  w_ptr, y_ptr, hy_ptr, cy_ptr);
      break;
    case rnn_enum::kGru:
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
                 int mode) {
  switch (mode) {
    case rnn_enum::kRnnRelu:
      break;
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      LstmBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                          input_size, state_size, x_ptr, hx_ptr, cx_ptr, w_ptr, y_ptr,
                          dy_ptr, dhy_ptr, dcy_ptr, dx_ptr, dhx_ptr, dcx_ptr, dw_ptr);
      break;
    case rnn_enum::kGru:
      break;
  }
}

template<typename DType>
class RNNOp {
 public:
  explicit RNNOp(RNNParam p) {
    param_ = p;
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(param_.mode, rnn_enum::kLstm) << "Only lstm mode is supported at the moment.";
    if (param_.bidirectional || param_.num_layers != 1) {
      LOG(FATAL) << "Only single layer and unidirectional is supported at the moment";
    }

    size_t in_expected = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    size_t out_expected = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    if (!param_.state_outputs) {
      out_expected = 1;
    }
    // the last output is used for training mode. It reserves forward intermediate result
    ++out_expected;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensor
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    DType* x_ptr = in_data[rnn_enum::kData].dptr<DType>();
    DType* w_ptr = in_data[rnn_enum::kParams].dptr<DType>();
    DType* hx_ptr = in_data[rnn_enum::kState].dptr<DType>();
    DType* y_ptr = out_data[rnn_enum::kOut].dptr<DType>();

    DType* hy_ptr = NULL;
    if (param_.state_outputs)
      hy_ptr = out_data[rnn_enum::kStateOut].dptr<DType>();

    DType* cx_ptr = NULL;
    DType* cy_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      cx_ptr = in_data[rnn_enum::kStateCell].dptr<DType>();
      if (param_.state_outputs) {
        cy_ptr = out_data[rnn_enum::kStateCellOut].dptr<DType>();
      }
    }

    // allocate temp space
    size_t workspace_size = GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                                                param_.state_size, param_.mode);
    Tensor<cpu, 1, DType> workspace = ctx.requested[rnn_enum::kTempSpace]
        .get_space_typed<cpu, 1, DType>(Shape1(workspace_size), s);
    int direction = param_.bidirectional ? 2 : 1;

    if (ctx.is_train) {
      DType* reserve_space_ptr = out_data[out_expected - 1].dptr<DType>();
      RNNForwardTraining<DType>(workspace.dptr_,
                                reserve_space_ptr,
                                param_.state_outputs,
                                param_.num_layers,
                                direction,
                                param_.seq_length_,
                                param_.batch_size_,
                                param_.input_size_,
                                param_.state_size,
                                x_ptr,
                                hx_ptr,
                                cx_ptr,
                                w_ptr,
                                y_ptr,
                                hy_ptr,
                                cy_ptr,
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
                                 x_ptr,
                                 hx_ptr,
                                 cx_ptr,
                                 w_ptr,
                                 y_ptr,
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
    CHECK_EQ(param_.mode, rnn_enum::kLstm) << "Only lstm mode is supported at the moment.";
    if (param_.bidirectional || param_.num_layers != 1) {
      LOG(FATAL) << "Only single layer and unidirectional is supported at the moment";
    }
    size_t in_expected = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    size_t out_expected = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    if (!param_.state_outputs) {
      out_expected = 1;
    }
    ++out_expected;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(out_grad.size(), out_expected - 1);
    CHECK_EQ(req.size(), in_expected);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo) << "AddTo is not supported for state";
    CHECK_NE(req[rnn_enum::kParams], kAddTo) << "AddTo is not supported for params";
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensors
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    DType* x_ptr = in_data[rnn_enum::kData].dptr<DType>();
    DType* w_ptr = in_data[rnn_enum::kParams].dptr<DType>();
    DType* hx_ptr = in_data[rnn_enum::kState].dptr<DType>();
    DType* y_ptr = out_data[rnn_enum::kOut].dptr<DType>();

    DType* dx_ptr = in_grad[rnn_enum::kData].dptr<DType>();
    DType* dw_ptr = in_grad[rnn_enum::kParams].dptr<DType>();
    DType* dhx_ptr = in_grad[rnn_enum::kState].dptr<DType>();
    DType* dy_ptr = out_grad[rnn_enum::kOut].dptr<DType>();

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
    // the last output is temp space that reserve forward intermediate result
    DType* reserve_space_ptr = out_data[out_expected - 1].dptr<DType>();

    // allocate temp space
    size_t workspace_size = GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                                                param_.state_size, param_.mode);
    Tensor<cpu, 1, DType> workspace = ctx.requested[rnn_enum::kTempSpace]
        .get_space_typed<cpu, 1, DType>(Shape1(workspace_size), s);

    int direction = param_.bidirectional ? 2 : 1;
    RNNBackward<DType>(workspace.dptr_,
                       reserve_space_ptr,
                       param_.num_layers,
                       direction,
                       param_.seq_length_,
                       param_.batch_size_,
                       param_.input_size_,
                       param_.state_size,
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
                       param_.mode);
  }

 private:
  RNNParam param_;
};  // class RNNOp

template<typename xpu>
void RNNCompute(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[rnn_enum::kData].type_flag_, DType, {
    RNNOp<DType> op(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void RNNGradCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> out_data{inputs[3]};
  std::vector<TBlob> out_grad{inputs[4]};

  int index = 5;
  if (param.state_outputs) {
    out_data.push_back(inputs[index++]);
    out_grad.push_back(inputs[index++]);
  }

  if (param.mode == rnn_enum::kLstm) {
    in_data.push_back(inputs[index++]);
    if (param.state_outputs) {
      out_data.push_back(inputs[index++]);
      out_grad.push_back(inputs[index++]);
    }
  }
  out_data.push_back(inputs[index]);
  const std::vector<TBlob> &in_grad = outputs;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    RNNOp<DType> op(param);
    op.Backward(ctx, out_grad, in_data, out_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RNN_INL_H_
