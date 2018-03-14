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
 * \author Sebastian Bodenstein, Shu Zhang(shu.zhang@intel.com)
*/
#include "./rnn-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(RNNParam);
static inline std::vector<std::string> ListArguments(const RNNParam& param_) {
  if (param_.mode == rnn_enum::kLstm) {
    return {"data", "parameters", "state", "state_cell"};
  } else {
    return {"data", "parameters", "state"};
  }
}
static inline int NumVisibleOutputs(const NodeAttrs& attrs) {
  const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
  int mode_num = (params.mode == rnn_enum::kLstm) ? 2 : 1;
  int num_outputs = params.state_outputs ? (mode_num + 1) : 1;
  return num_outputs;
}
static bool RNNShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_shape,
                     std::vector<TShape> *out_shape) {
  const RNNParam& param_ = nnvm::get<RNNParam>(attrs.parsed);
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
  if (param_.state_outputs) {
    // outStateShape: [layer_num, batch, state size]
    TShape outStateShape = dshape;
    outStateShape[0] = total_layers;
    outStateShape[1] = batch_size;
    outStateShape[2] = param_.state_size;
    out_shape->push_back(outStateShape);
    // Deal with lstm cell state
    if (param_.mode == rnn_enum::kLstm)
      out_shape->push_back(outStateShape);
  }
  // the reserve space shape
  TShape outReserveShape = (*in_shape)[rnn_enum::kParams];
  outReserveShape[0] = GetRNNReserveSpaceSize(dshape[0],
                                              batch_size,
                                              param_.state_size,
                                              param_.mode);
  out_shape->push_back(outReserveShape);
  return true;
}

static bool RNNType(const nnvm::NodeAttrs& attrs,
                    std::vector<int> *in_type,
                    std::vector<int> *out_type) {
  const RNNParam& param_ = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (index_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  if (param_.state_outputs) {
    out_type->push_back(dtype);
    // Deal with lstm cell state
    if (param_.mode == rnn_enum::kLstm)
      out_type->push_back(dtype);
  }
  out_type->push_back(dtype);
  return true;
}

inline static bool RNNStorageType(const nnvm::NodeAttrs& attrs,
                                  const int dev_mask,
                                  DispatchMode* dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  DispatchMode wanted_mode = DispatchMode::kFCompute;
  return storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                             dispatch_mode, wanted_mode);
}

inline static bool BackwardRNNStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs) {
  DispatchMode wanted_mode = DispatchMode::kFCompute;
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
    heads.emplace_back(nnvm::NodeEntry{n, rnn_enum::kOut, 0});
    heads.push_back(ograd[rnn_enum::kOut]);
    // index of space that reserve forward intermediate result
    uint32_t kTmpSpaceIdx = rnn_enum::kOut + 1;
    if (params.state_outputs) {
      heads.emplace_back(nnvm::NodeEntry{n, rnn_enum::kStateOut, 0});
      heads.push_back(ograd[rnn_enum::kStateOut]);
      ++kTmpSpaceIdx;
    }
    if (params.mode == rnn_enum::kLstm) {
      heads.push_back(n->inputs[rnn_enum::kStateCell]);
      if (params.state_outputs) {
        heads.emplace_back(nnvm::NodeEntry{n, rnn_enum::kStateCellOut, 0});
        heads.push_back(ograd[rnn_enum::kStateCellOut]);
        ++kTmpSpaceIdx;
      }
    }
    heads.emplace_back(nnvm::NodeEntry{n, kTmpSpaceIdx, 0});
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

NNVM_REGISTER_OP(RNN)
.describe(R"code(Applies a recurrent layer to input
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<RNNParam>)
.set_num_inputs(4)
.set_num_inputs([](const NodeAttrs& attrs) {
    const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
    return params.mode == rnn_enum::kLstm ? 4 : 3;
})
.set_num_outputs([](const NodeAttrs& attrs) {
    return NumVisibleOutputs(attrs) + 1;
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
    return NumVisibleOutputs(attrs);
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
    const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
    return ListArguments(params);
})
.set_attr<nnvm::FInferShape>("FInferShape", RNNShape)
.set_attr<nnvm::FInferType>("FInferType", RNNType)
.set_attr<FInferStorageType>("FInferStorageType", RNNStorageType)
.set_attr<FCompute>("FCompute<cpu>", RNNCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", RNNGrad{"_backward_RNN"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to RNN")
.add_argument("parameters", "NDArray-or-Symbol",
              "Vector of all RNN trainable parameters concatenated")
.add_argument("state", "NDArray-or-Symbol", "initial hidden state of the RNN")
.add_argument("state_cell", "NDArray-or-Symbol",
              "initial cell state for LSTM networks (only for LSTM)")
.add_arguments(RNNParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_RNN)
.set_num_outputs([](const NodeAttrs& attrs) {
    const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
    return params.mode == rnn_enum::kLstm ? 4 : 3;
})
.set_attr_parser(ParamParser<RNNParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BackwardRNNStorageType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", RNNGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
