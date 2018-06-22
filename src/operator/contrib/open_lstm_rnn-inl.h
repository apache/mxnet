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
 * Copyright (c) 2018 by Contributors
 * \file open_lstm_rnn-inl.h
 * \brief LSTM RNN Open-Source CUDA Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko
 */
#ifndef MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INL_H_
#define MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace open_lstm_rnn_enum {
  enum OpenLSTMRNNOpInputs {kData, kInitHidden, kInitCell,
                            ki2hWeight, ki2hBias,
                            kh2hWeight, kh2hBias};
  enum OpenLSTMRNNOpOutputs {kConcatHiddenStates,
                             kFinalHidden, kFinalCell};
}

struct OpenLSTMRNNParam : public dmlc::Parameter < OpenLSTMRNNParam > {
  // parameters that determine RNN configurations
  uint32_t num_layers, num_hidden_units;
  float i_dp_prob;  // input dropout probability
  // parameters that are inferred from input data
  unsigned batch_size, seq_len, embed_dim;

  bool state_outputs;  // whether to output the final hidden and cell state

  DMLC_DECLARE_PARAMETER(OpenLSTMRNNParam) {
    DMLC_DECLARE_FIELD(num_layers)
    .describe("number of stacked layers");
    DMLC_DECLARE_FIELD(num_hidden_units)
    .describe("number of hidden units");
    DMLC_DECLARE_FIELD(i_dp_prob).set_default(0.).set_range(0, 1)
    .describe("input dropout probability");

    DMLC_DECLARE_FIELD(state_outputs).set_default(false)
    .describe("Whether to have final hidden and cell states as symbol outputs");
  }
};

template < typename xpu, typename DType >
class OpenLSTMRNNOp : public Operator {
 public:
  explicit OpenLSTMRNNOp(OpenLSTMRNNParam p)
  {}
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    // using namespace mshadow;
    // using namespace mshadow::expr;
    // OpenLSTMRNN can only run on the GPU
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    // using namespace mshadow;
    // using namespace mshadow::expr;
    // OpenLSTMRNN can only run on the GPU
  }
};

template < typename xpu >
Operator * CreateOp(OpenLSTMRNNParam param, int dtype);

#if DMLC_USE_CXX11
class OpenLSTMRNNProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "init_hidden", "init_cell",
            "i2h_weight", "i2h_bias",
            "h2h_weight", "h2h_bias"};
  }
  std::vector<std::string> ListOutputs() const override {
    if (param_.state_outputs)
      return {"concat_hidden_states",
              "final_hidden", "final_cell"};
    else
      return {"concat_hidden_states"};
  }

  int NumOutputs() const override {
    if (param_.state_outputs)
      return 3;
    else
      return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 7U) << "Input: [data, init_hidden, init_cell, "
                                      "i2h_weight, i2h_bias, "
                                      "h2h_weight, h2h_bias]";
    const TShape & dshape = (*in_shape)[open_lstm_rnn_enum::kData];
    CHECK_EQ(dshape.ndim(), 3U) << "Input data should be rank-3 tensor of dimension "
                                   "[batch_size, seq_len, embed_dim]";
    unsigned batch_size = dshape[0], embed_dim = dshape[2],
             num_layers = param_.num_layers,
             num_hidden_units = param_.num_hidden_units;
    SHAPE_ASSIGN_CHECK(*in_shape, open_lstm_rnn_enum::kInitHidden,
                       Shape3(num_layers, batch_size, num_hidden_units));
    SHAPE_ASSIGN_CHECK(*in_shape, open_lstm_rnn_enum::kInitCell,
                       Shape3(num_layers, batch_size, num_hidden_units));
    SHAPE_ASSIGN_CHECK(*in_shape, open_lstm_rnn_enum::ki2hWeight,
                       Shape1(4 * num_hidden_units * embed_dim +
                              (num_layers-1) * 4 * num_hidden_units * num_hidden_units));
    SHAPE_ASSIGN_CHECK(*in_shape, open_lstm_rnn_enum::ki2hBias,
                       Shape1(num_layers * 4 * num_hidden_units));
    SHAPE_ASSIGN_CHECK(*in_shape, open_lstm_rnn_enum::kh2hWeight,
                       Shape1(num_layers * 4 * num_hidden_units * num_hidden_units));
    SHAPE_ASSIGN_CHECK(*in_shape, open_lstm_rnn_enum::kh2hBias,
                       Shape1(num_layers * 4 * num_hidden_units));
    out_shape->clear();
    // oshape: [batch_size x seq_len x embed_dim]
    TShape oshape = dshape; oshape[2] = num_hidden_units;
    out_shape->push_back(oshape);  // concatenated hidden states
    if (param_.state_outputs) {
      oshape[0] = num_layers;
      oshape[1] = batch_size;
      oshape[2] = num_hidden_units;
      out_shape->push_back(oshape);  // final hidden state
      out_shape->push_back(oshape);  // final cell state
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if (((*in_type)[i] == -1))
        (*in_type)[i] = dtype;
      else
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. " <<
                                          "Expected " << dtype << " v.s. given " <<
                                          (*in_type)[i] << " at " << ListArguments()[i];
    }
    out_type->clear();
    out_type->push_back(dtype);  // concatenated hidden states
    if (param_.state_outputs) {
      out_type->push_back(dtype);
      out_type->push_back(dtype);
    }
    return true;
  }

  OperatorProperty * Copy() const override {
    auto ptr = new OpenLSTMRNNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "OpenLSTMRNN";
  }

  std::vector<int> DeclareBackwardDependency(const std::vector<int> &out_grad,
                                             const std::vector<int> &in_data,
                                             const std::vector<int> &out_data) const override {
    std::vector<int> dep = {in_data[open_lstm_rnn_enum::kData],
                            in_data[open_lstm_rnn_enum::kInitHidden],
                            in_data[open_lstm_rnn_enum::kInitCell],
                            in_data[open_lstm_rnn_enum::ki2hWeight],
                            in_data[open_lstm_rnn_enum::ki2hBias],
                            in_data[open_lstm_rnn_enum::kh2hWeight],
                            in_data[open_lstm_rnn_enum::kh2hBias],
                            out_data[open_lstm_rnn_enum::kConcatHiddenStates],
                            out_grad[open_lstm_rnn_enum::kConcatHiddenStates]};
    if (param_.state_outputs) {
      dep.push_back(out_data[open_lstm_rnn_enum::kFinalHidden]);
      dep.push_back(out_grad[open_lstm_rnn_enum::kFinalHidden]);
      dep.push_back(out_data[open_lstm_rnn_enum::kFinalCell]);
      dep.push_back(out_grad[open_lstm_rnn_enum::kFinalCell]);
    }
    return dep;
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {};
  }

  Operator * CreateOperator(Context ctx) const override {
    LOG(FATAL) << "OpenLSTMRNN can only run on the GPU.";
    return NULL;
  }

  Operator * CreateOperatorEx(Context ctx,
                              std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
  OpenLSTMRNNParam param_;
};

#endif

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INL_H_
