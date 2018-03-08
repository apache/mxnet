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
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include <mkl.h>
#include <mxnet/storage.h>
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
      size = (seq_length + 1) * batch_size * hidden_size * 4 + batch_size * hidden_size; //lstm
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
      size = seq_length * batch_size * hidden_size * 6; //lstm
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
    CHECK_EQ(param_.mode, rnn_enum::kLstm) << "Only lstm mode is supported at the moment.";

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
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];
    
    //allocate temp space
    size_t workspace_size = GetRNNWorkspaceSize(param_.seq_length_,
                         param_.batch_size_, param_.state_size, param_.mode);
    Tensor<cpu, 1, DType> workspace = ctx.requested[rnn_enum::kTempSpace]
                .get_space_typed<cpu, 1, DType>(Shape1(workspace_size), s);
    int direction = param_.bidirectional ? 2 : 1;
    if (ctx.is_train) {
      size_t r_size = GetRNNReserveSpaceSize(param_.seq_length_,
                       param_.batch_size_, param_.state_size, param_.mode);
      if (init_space_ && reserve_space_size_ < r_size) {
        Storage::Get()->Free(reserve_space_);
        init_space_ = false;
        reserve_space_size_ = r_size;
      }
      if (!init_space_) {
        reserve_space_ = Storage::Get()->Alloc(
                         reserve_space_size_ * sizeof(DType), Context::CPU());
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
                                x_ptr,
                                hx_ptr,
                                cx_ptr,
                                w_ptr,
                                y_ptr,
                                hy_ptr,
                                cy_ptr);
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
                                 cy_ptr);
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
    CHECK_NE(req[rnn_enum::kParams], kAddTo) << "AddTo is not supported for params";
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensors
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
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
    
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];
    
    //allocate temp space
    size_t workspace_size = GetRNNWorkspaceSize(param_.seq_length_,
                         param_.batch_size_, param_.state_size, param_.mode);
    Tensor<cpu, 1, DType> workspace = ctx.requested[rnn_enum::kTempSpace]
                .get_space_typed<cpu, 1, DType>(Shape1(workspace_size), s);
    
    int direction = param_.bidirectional ? 2 : 1;
    size_t r_size = GetRNNReserveSpaceSize(param_.seq_length_,
                     param_.batch_size_, param_.state_size, param_.mode);
    if (init_space_ && reserve_space_size_ < r_size) {
      Storage::Get()->Free(reserve_space_);
      init_space_ = false;
      reserve_space_size_ = r_size;
    }
    if (!init_space_) {
      reserve_space_ = Storage::Get()->Alloc(
                       reserve_space_size_ * sizeof(DType), Context::CPU());
    }
    DType* reserve_space_ptr = static_cast<DType*>(reserve_space_.dptr);
    RNNBackward<DType>(workspace.dptr_, 
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
                       dy_ptr,
                       dhy_ptr,
                       dcy_ptr,
                       dx_ptr,
                       dhx_ptr,
                       dcx_ptr,
                       dw_ptr);
  }

 private:
  RNNParam param_;
  bool init_space_;
  size_t reserve_space_size_;
  Storage::Handle reserve_space_;
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
      out_grad.push_back(inputs[index]);
    }
  }
  const std::vector<TBlob> &in_grad = outputs;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    RNNOp<DType> op(param);
    op.Backward(ctx, out_grad, in_data, out_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RNN_INL_H_
