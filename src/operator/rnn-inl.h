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
#include "./operator_common.h"

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
    .describe("Fraction of the input that gets dropped out at training time");

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
      CHECK_EQ(in_shape->size(), 4) << "Input:[data, parameters, state, cell_state]";
    } else {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, parameters, state]";
    }
    const TShape &dshape = (*in_shape)[rnn_enum::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 3) \
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
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
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
