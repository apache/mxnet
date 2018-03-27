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

inline int GetRnnParamSize(int num_layer,
                           int input_size,
                           int state_size,
                           int direction,
                           int mode) {
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
  int param_size = size1 + (num_layer - 1) * size2;
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
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kGru:
      size = seq_length * batch_size * hidden_size * 4 + batch_size * hidden_size * 6;
      break;
    case rnn_enum::kLstm:
      LOG(FATAL) << "Only GRU is supported at the moment";
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
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
    case rnn_enum::kRnnTanh:
    case rnn_enum::kGru:
      size = seq_length * batch_size * hidden_size * 5 + batch_size * hidden_size * 7 +
          2 * seq_length * batch_size * 3 * hidden_size;
      break;
    case rnn_enum::kLstm:
      LOG(FATAL) << "Only GRU is supported at the moment";
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

  bool operator==(const RNNParam& other) const {
    return this->state_size == other.state_size &&
           this->num_layers == other.num_layers &&
           this->bidirectional == other.bidirectional &&
           this->state_outputs == other.state_outputs &&
           this->mode == other.mode &&
           this->seq_length_ == other.seq_length_ &&
           this->batch_size_ == other.batch_size_ &&
           this->input_size_ == other.input_size_ &&
           this->lstm_q_ == other.lstm_q_;
  }
};

typedef ParamOpSign<RNNParam> RNNSignature;

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
    case rnn_enum::kRnnTanh:
    case rnn_enum::kGru:
      GruForwardTraining<DType>(rs, state_outputs, num_layers, direction, seq_length,
                                batch_size, input_size, state_size, x_ptr, hx_ptr,
                                w_ptr, y_ptr, hy_ptr);
      break;
    case rnn_enum::kLstm:
      LOG(FATAL) << "Only GRU is supported at the moment";
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
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kGru:
      GruForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                 batch_size, input_size, state_size, x_ptr, hx_ptr,
                                 w_ptr, y_ptr, hy_ptr);
      break;
    case rnn_enum::kLstm:
      LOG(FATAL) << "Only GRU is supported at the moment";
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
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
      LOG(FATAL) << "Only GRU is supported at the moment";
      break;
    case rnn_enum::kGru:
      GruBackward<DType>(rs, num_layers, direction, seq_length, batch_size,
                         input_size, state_size, x_ptr, hx_ptr, w_ptr,
                         dy_ptr, dhy_ptr, dx_ptr, dhx_ptr, dw_ptr);
      break;
  }
}

template<typename DType>
class RNNOp {
 public:
  explicit RNNOp(RNNParam p) {
    param_ = p;
    init_space_ = false;
    reserve_space_size_ = 0;
  }

  ~RNNOp() {
    if (init_space_) {
      Storage::Get()->Free(reserve_space_);
    }
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.mode == rnn_enum::kGru) << "Only gru mode is supported at the moment.";

    size_t in_expected = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    size_t out_expected = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    if (!param_.state_outputs) {
      out_expected = 1;
    }
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
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
      const size_t r_size = GetRNNReserveSpaceSize(param_.seq_length_, param_.batch_size_,
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
                                y.dptr_,
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
    CHECK(param_.mode == rnn_enum::kGru) << "Only gru mode is supported at the moment.";
    if (param_.bidirectional || param_.num_layers != 1) {
      LOG(FATAL) << "Only single layer and unidirectional is supported at the moment";
    }
    size_t in_expected = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    size_t out_expected = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    if (!param_.state_outputs) {
      out_expected = 1;
    }
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(out_grad.size(), out_expected);
    CHECK_EQ(req.size(), in_expected);
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

    size_t r_size = GetRNNReserveSpaceSize(param_.seq_length_, param_.batch_size_,
                                           param_.state_size, param_.mode);
    if (!init_space_ || reserve_space_size_ != r_size) {
      LOG(FATAL) << " Check forward init error" << reserve_space_size_;
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
                       param_.mode);
  }

 private:
  RNNParam param_;
  bool init_space_;
  size_t reserve_space_size_;
  Storage::Handle reserve_space_;
};  // class RNNOp

template<typename DType>
static RNNOp<DType> &GetRNNOp(const RNNParam &param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local RNNOp<DType> op(param);
#else
  static MX_THREAD_LOCAL RNNOp<DType> op(param);
#endif
  return op;
}

template<typename xpu>
void RNNCompute(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[rnn_enum::kData].type_flag_, DType, {
    GetRNNOp<DType>(param).Forward(ctx, inputs, req, outputs);
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
      out_grad.push_back(inputs[index]);
    }
  }
  const std::vector<TBlob> &in_grad = outputs;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    GetRNNOp<DType>(param).Backward(ctx, out_grad, in_data, out_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::RNNParam> {
  size_t operator()(const mxnet::op::RNNParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.state_size);
    ret = dmlc::HashCombine(ret, val.num_layers);
    ret = dmlc::HashCombine(ret, val.bidirectional);
    ret = dmlc::HashCombine(ret, val.state_outputs);
    ret = dmlc::HashCombine(ret, val.mode);
    ret = dmlc::HashCombine(ret, val.seq_length_);
    ret = dmlc::HashCombine(ret, val.batch_size_);
    ret = dmlc::HashCombine(ret, val.input_size_);
    ret = dmlc::HashCombine(ret, val.lstm_q_);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_RNN_INL_H_
