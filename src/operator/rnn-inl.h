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
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_RNN_INL_H_
#define MXNET_OPERATOR_RNN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./linalg.h"

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

template<typename xpu, typename DType>
class RNNOp : public Operator {
 public:
  explicit RNNOp(RNNParam p) {
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(sbodenstein): add MShadow implementation
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(sbodenstein): add MShadow implementation
  }

 private:
  RNNParam param_;
};  // class RNNOp

template<typename DType>
class RNNOp<cpu, DType> : public Operator {
 public:
  explicit RNNOp(RNNParam param) {
    this->param_ = param;
    // RNN Mode
    param_.lstm_q_ = false;
    switch (param_.mode) {
      case rnn_enum::kLstm:
        param_.lstm_q_ = true;
        break;
      default:
        LOG(FATAL) << "only LSTM is implmented on CPU";
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    // Layout TNC
    CHECK(!ctx.is_train) << "only inference mode is available"
      "for cpu at the moment.";
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;

    if (!param_.state_outputs)
      LOG(FATAL) << "no state outputs is currently not supported for cpu.";

    CHECK_EQ(req[rnn_enum::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);

    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensors
    // w layout i2h_w, h2h_w, i2h_b, h2h_b
    Tensor<cpu, 3, DType> x =
        in_data[rnn_enum::kData].get<cpu, 3, DType>(s);  // TNC
    Tensor<cpu, 1, DType> w = in_data[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> hx =
        in_data[rnn_enum::kState].get<cpu, 3, DType>(s);  // LNC
    Tensor<cpu, 3, DType> y =
        out_data[rnn_enum::kOut].get<cpu, 3, DType>(s);  // TNC
    int64_t seq_len = x.shape_[0];
    int64_t num_layers = hx.shape_[0];
    int64_t batch_size = x.shape_[1];
    int64_t h_channel = hx.shape_[2];
    int64_t in_channel = x.shape_[2];
    Tensor<cpu, 2, DType> x_flatten = in_data[rnn_enum::kData]
      .get_with_shape<cpu, 2, DType>(
          mshadow::Shape2(seq_len * batch_size, in_channel), s);  // (T*N)C
    Tensor<cpu, 2, DType> y_flatten = out_data[rnn_enum::kOut]
      .get_with_shape<cpu, 2, DType>(
          mshadow::Shape2(
              y.shape_[0] * y.shape_[1], y.shape_[2]), s);  // (T*N)C

    CHECK(x.CheckContiguous());
    CHECK(w.CheckContiguous());
    CHECK(hx.CheckContiguous());
    CHECK(y.CheckContiguous());

    if (param_.lstm_q_) {
      const size_t kNumMat = 4;
      int64_t fused_h_ch = kNumMat * h_channel;
      int64_t h_size = batch_size * fused_h_ch;
      int64_t num_dir = 1 + param_.bidirectional;
      int64_t h2h_w_size = h_channel * fused_h_ch;

      Tensor<cpu, 3, DType> cx =
          in_data[rnn_enum::kStateCell].get<cpu, 3, DType>(s);
      CHECK(cx.CheckContiguous());

      Tensor<cpu, 3, DType> cy =
          out_data[rnn_enum::kStateCellOut].get<cpu, 3, DType>(s);
      Tensor<cpu, 3, DType> hy =
          out_data[rnn_enum::kStateOut].get<cpu, 3, DType>(s);
      CHECK(cy.CheckContiguous());
      CHECK(hy.CheckContiguous());

      DType* workspace_addr =
      static_cast<DType *>(ctx.requested[rnn_enum::kTempSpace]
          .get_host_space_internal(sizeof(DType) *
                                  (seq_len * h_size + h_size
                                  + y.shape_[0] * y.shape_[1] * y.shape_[2])));
      Tensor<cpu, 3, DType> i2h_y(
          workspace_addr, mshadow::Shape3(seq_len, batch_size, fused_h_ch));
      Tensor<cpu, 2, DType> i2h_y_flatten(
          workspace_addr, mshadow::Shape2(seq_len * batch_size, fused_h_ch));
      Tensor<cpu, 2, DType> h2h_y(workspace_addr
          + seq_len * h_size, mshadow::Shape2(batch_size, fused_h_ch));
      Tensor<cpu, 3, DType> y_tmp(workspace_addr
          + (seq_len + 1) * h_size, y.shape_);
      Tensor<cpu, 2, DType> y_flatten_tmp(workspace_addr
          + (seq_len + 1) * h_size, y_flatten.shape_);
      CHECK(i2h_y.CheckContiguous());
      CHECK(h2h_y.CheckContiguous());
      CHECK(y_tmp.CheckContiguous());

      for (int64_t layer = 0; layer < num_layers; layer++) {
        int reverse_dir = 0;
        int out_tmp = 0;
        if (param_.bidirectional && layer % 2)
          reverse_dir = 1;
        if (layer / num_dir % 2 == 0)
          out_tmp = 1;
        mshadow::Shape<2> i2h_w_shape = mshadow::Shape2(fused_h_ch,
            (layer < num_dir) ? in_channel : num_dir * h_channel);
        mshadow::Shape<2> h2h_w_shape = mshadow::Shape2(fused_h_ch, h_channel);
        int64_t start = layer < num_dir ?
            (layer * (in_channel * fused_h_ch + h2h_w_size)) :  // input layer
              (num_dir * (in_channel * fused_h_ch + h2h_w_size)
              + (layer - num_dir) * (h2h_w_size * num_dir + h2h_w_size));
        Tensor<cpu, 2, DType> i2h_w(w.dptr_ + start, i2h_w_shape);
        start += layer < num_dir ?
            in_channel * fused_h_ch : h2h_w_size * num_dir;
        Tensor<cpu, 2, DType> h2h_w(w.dptr_ + start, h2h_w_shape);
        start = num_dir * (in_channel * fused_h_ch + h2h_w_size)
            + (num_layers - num_dir) * (h2h_w_size * (num_dir + 1))
              + layer * fused_h_ch * 2;
        Tensor<cpu, 1, DType> i2h_b = w.Slice(start, start + fused_h_ch);
        start += fused_h_ch;
        Tensor<cpu, 1, DType> h2h_b = w.Slice(start, start + fused_h_ch);
        if (out_tmp) {
          linalg_gemm(layer < num_dir ? x_flatten:y_flatten, i2h_w,
              i2h_y_flatten, false, true, s);
        } else {
          linalg_gemm(layer < num_dir ? x_flatten:y_flatten_tmp, i2h_w,
              i2h_y_flatten, false, true, s);
        }
        i2h_y_flatten += repmat(i2h_b, seq_len * batch_size);
        for (int64_t t = 0; t < seq_len; t++) {
          int64_t timestep = t;
          if (reverse_dir)
            timestep = seq_len - 1 - t;
          linalg_gemm(t == 0 ? hx[layer]:hy[layer], h2h_w, h2h_y,
              false, true, s);
          h2h_y += repmat(h2h_b, batch_size);
          // fused element-wise ops
          LSTMFusedElementWiseCPUOps(i2h_y[timestep], cx[layer], h2h_y,
              y[timestep], out_tmp ? y_tmp[timestep]: y[timestep],
                hy[layer], cy[layer], batch_size, h_channel, t,
                reverse_dir, out_tmp && (layer == num_layers - 1));
        }
      }
    } else {
      LOG(FATAL) << "only LSTM is available for cpu at the moment.";
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
      const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    LOG(FATAL) << "LSTM backward is not available for cpu at the moment.";
  }

 private:
  RNNParam param_;

  void LSTMFusedElementWiseCPUOps(const Tensor<cpu, 2, DType> &i2h_y,
                                  const Tensor<cpu, 2, DType> &cx,
                                  const Tensor<cpu, 2, DType> &h2h_y,
                                  const Tensor<cpu, 2, DType> &y,
                                  // holding intermediate layer output
                                  const Tensor<cpu, 2, DType> &tmp,
                                  const Tensor<cpu, 2, DType> &hy,
                                  const Tensor<cpu, 2, DType> &cy,
                                  const int64_t batch_size,
                                  const int64_t h_channel,
                                  const int64_t t,
                                  const int reverse_dir,
                                  const int copy_tmp2y) {
    int64_t length = batch_size * h_channel;
    #pragma omp parallel for
    for (int64_t ji = 0; ji < length; ++ji) {
      int64_t j = ji / h_channel;  // batch dim
      int64_t i = ji % h_channel;
      int64_t f = i + h_channel;
      int64_t c = i + h_channel * 2;
      int64_t o = i + h_channel * 3;
      int64_t j_pos = j * h_channel * 4;
      h2h_y.dptr_[j_pos + i] += i2h_y.dptr_[j_pos + i];
      h2h_y.dptr_[j_pos + f] += i2h_y.dptr_[j_pos + f];
      h2h_y.dptr_[j_pos + o] += i2h_y.dptr_[j_pos + o];
      h2h_y.dptr_[j_pos + c] += i2h_y.dptr_[j_pos + c];
      h2h_y.dptr_[j_pos + i] = 1.0f / (1.0f + math::exp(-h2h_y.dptr_[j_pos + i]));
      h2h_y.dptr_[j_pos + f] = 1.0f / (1.0f + math::exp(-h2h_y.dptr_[j_pos + f]));
      h2h_y.dptr_[j_pos + o] = 1.0f / (1.0f + math::exp(-h2h_y.dptr_[j_pos + o]));
      h2h_y.dptr_[j_pos + c] = tanh(h2h_y.dptr_[j_pos + c]);
      cy[j][i] = h2h_y.dptr_[j_pos + f] * (t == 0 ? cx[j][i]:cy[j][i])
          + h2h_y.dptr_[j_pos + i] * h2h_y.dptr_[j_pos + c];
      hy[j][i] = h2h_y.dptr_[j_pos + o] * tanh(cy[j][i]);
      tmp[j][i + h_channel * reverse_dir] = hy[j][i];
      if (copy_tmp2y) {
        y[j][i] = tmp[j][i];
        if (reverse_dir)
          y[j][i + h_channel] = tmp[j][i + h_channel];
      }
    }
  }
};  // class RNNOp

template<typename xpu>
Operator* CreateOp(RNNParam param, int dtype);

#if DMLC_USE_CXX11
class RNNProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (param_.mode == rnn_enum::kLstm) {
      return {"data", "parameters", "state", "state_cell"};
    } else {
      return {"data", "parameters", "state"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> outputs = {"output"};
    if (!param_.state_outputs)
      return outputs;
    else
      outputs.push_back("state");
    if (param_.mode == rnn_enum::kLstm)
      outputs.push_back("state_cell");
    return outputs;
  }

  int NumOutputs() const override {
    int mode_num = (param_.mode == rnn_enum::kLstm) ? 2 : 1;
    int num_outputs = param_.state_outputs ? (mode_num + 1) : 1;
    return num_outputs;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.mode == rnn_enum::kLstm) {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, parameters, state, cell_state]";
    } else {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, parameters, state]";
    }
    const TShape &dshape = (*in_shape)[rnn_enum::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 3U) \
        << "Input data should be rank-3 tensor of dim [sequence length, batch size, input size]";
    // data: [sequence len, batch, input dimension]
    int batch_size = dshape[1];
    int input_size = dshape[2];
    int numDirections = param_.bidirectional ? 2 : 1;
    int total_layers = numDirections * param_.num_layers;  // double for bidirectional
    SHAPE_ASSIGN_CHECK(*in_shape,
                       rnn_enum::kState,
                       Shape3(total_layers, batch_size, param_.state_size));
    if (param_.mode == rnn_enum::kLstm)
      SHAPE_ASSIGN_CHECK(*in_shape,
                        rnn_enum::kStateCell,
                        Shape3(total_layers, batch_size, param_.state_size));

    // calculate parameter vector length
    int param_size = rnn_param_size(param_.num_layers,
                                    input_size,
                                    param_.state_size,
                                    param_.bidirectional,
                                    param_.mode);
    SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kParams, Shape1(param_size));

    out_shape->clear();
    // output: [sequence len, batch, output size]
    TShape oshape = dshape;
    oshape[2] = numDirections * param_.state_size;
    out_shape->push_back(oshape);
    if (!param_.state_outputs) {
      return true;
    } else {
      // outStateShape: [layer_num, batch, state size]
      TShape outStateShape = dshape;
      outStateShape[0] = total_layers;
      outStateShape[1] = batch_size;
      outStateShape[2] = param_.state_size;
      out_shape->push_back(outStateShape);
      // Deal with lstm cell state
      if (param_.mode == rnn_enum::kLstm)
        out_shape->push_back(outStateShape);
      return true;
    }
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    if (!param_.state_outputs) {
      return true;
    } else {
      out_type->push_back(dtype);
      // Deal with lstm cell state
      if (param_.mode == rnn_enum::kLstm)
        out_type->push_back(dtype);
      return true;
    }
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RNNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "RNN";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> dep = {in_data[rnn_enum::kData], in_data[rnn_enum::kParams],
        in_data[rnn_enum::kState], out_data[rnn_enum::kOut], out_grad[rnn_enum::kOut]};

    if (param_.state_outputs) {
      dep.push_back(out_data[rnn_enum::kStateOut]);
      dep.push_back(out_grad[rnn_enum::kStateOut]);
    }

    if (param_.mode == rnn_enum::kLstm) {
      dep.push_back(in_data[rnn_enum::kStateCell]);
      if (param_.state_outputs) {
        dep.push_back(out_data[rnn_enum::kStateCellOut]);
        dep.push_back(out_grad[rnn_enum::kStateCellOut]);
      }
    }
    return dep;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  RNNParam param_;
};  // class RNNProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RNN_INL_H_
