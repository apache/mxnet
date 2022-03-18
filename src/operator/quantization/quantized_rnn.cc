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
 * \file quantized_rnn.cc
 * \brief Common functions for quantized recurrent neural network
 * \author Zixuan Wei
 */

#include <dmlc/logging.h>
#include <string>
#include <utility>
#include <vector>

#include "operator/rnn-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/quantization/quantized_rnn-inl.h"

#if MXNET_USE_ONEDNN == 1
#include "operator/quantization/dnnl/dnnl_quantized_rnn-inl.h"
#endif

namespace mxnet {
namespace op {

uint32_t QuantizedRnnNumInputs(const NodeAttrs& attrs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm)
      << "Quantized recurrent neural network only supports LSTM operator on "
         "CPU.";
  return 6U;
}

uint32_t QuantizedRnnNumOutputs(const NodeAttrs& attrs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm)
      << "Quantized recurrent neural network only supports LSTM operator on "
         "CPU.";
  return param.state_outputs ? 3U : 1U;
}

std::vector<std::string> QuantizedRnnInputNames(const NodeAttrs& attrs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm)
      << "Quantized recurrent neural network only supports LSTM operator on "
         "CPU.";
  return std::vector<std::string>{
      "data", "parameters", "state", "state_cell", "min_data", "max_data"};
}

std::vector<std::string> QuantizedRnnOutputNames(const NodeAttrs& attrs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm)
      << "Quantized recurrent neural network only supports LSTM operator on "
         "CPU.";
  if (param.state_outputs) {
    return std::vector<std::string>{"output", "state_output", "statecell_ouput"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

bool QuantizedRnnShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape>* in_shape,
                       std::vector<TShape>* out_shape) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm) << "Quantized RNN operator only supports LSTM mode.";

  const uint32_t num_inputs  = QuantizedRnnNumInputs(attrs);
  const uint32_t num_outputs = QuantizedRnnNumOutputs(attrs);
  CHECK_EQ(in_shape->size(), num_inputs)
      << "Arguments' size of quantized RNN operator is mismatched. Expected " << num_inputs
      << " argmuments but got " << in_shape->size() << ".";
  CHECK_EQ(out_shape->size(), num_outputs);

  const mxnet::TShape dshape = in_shape->at(quantized_rnn::kData);
  if (!mxnet::ndim_is_known(dshape))
    return false;
  CHECK_EQ(dshape.ndim(), 3U) << "Input data of RNN operator should be 3-rank "
                                 "tensor of dim [steps, batch, input size]";
  const dim_t batch_size = dshape[1];
  const dim_t input_size = dshape[2];
  const dim_t directions = param.bidirectional ? 2 : 1;
  const dim_t total_lyrs = directions * param.num_layers;
  const dim_t state_size = param.state_size;
  SHAPE_ASSIGN_CHECK(*in_shape, quantized_rnn::kState, Shape3(total_lyrs, batch_size, state_size));
  if (param.mode == rnn_enum::kLstm)
    SHAPE_ASSIGN_CHECK(
        *in_shape, quantized_rnn::kStateCell, Shape3(total_lyrs, batch_size, state_size));

  const int param_size_fp = GetRnnParamSize(
      param.num_layers, input_size, state_size, directions, param.mode, param.projection_size);
  SHAPE_ASSIGN_CHECK(*in_shape, quantized_rnn::kParams, Shape1(param_size_fp));
  const uint32_t num_base_inputs = GetRnnNumInputs(param);
  for (size_t i = num_base_inputs; i < num_inputs; ++i)
    SHAPE_ASSIGN_CHECK(*in_shape, i, Shape1(1));

  out_shape->clear();
  out_shape->push_back({dshape[0], batch_size, directions * state_size});  // output dim: [T, N, C]
  if (param.state_outputs) {
    out_shape->push_back({total_lyrs, batch_size, state_size});  // state dim: [L*D, N, C]
    if (param.mode == rnn_enum::kLstm)
      out_shape->push_back({total_lyrs, batch_size, state_size});  // cell dim: [L*D, N, C]
  }
  return true;
}

bool QuantizedRnnType(const nnvm::NodeAttrs& attrs,
                      std::vector<int>* in_type,
                      std::vector<int>* out_type) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm) << "Quantized RNN operator only supports LSTM mode.";

  const uint32_t num_inputs  = QuantizedRnnNumInputs(attrs);
  const uint32_t num_outputs = QuantizedRnnNumOutputs(attrs);
  CHECK_EQ(in_type->size(), num_inputs);
  CHECK_EQ(out_type->size(), num_outputs);

  CHECK_EQ(in_type->at(quantized_rnn::kData), mshadow::kUint8)
      << "Quantized RNN operator only supports uint8 input, while "
      << in_type->at(quantized_rnn::kData) << " is given.";
  TYPE_ASSIGN_CHECK(*in_type, quantized_rnn::kParams, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, quantized_rnn::kState, mshadow::kFloat32);
  const uint32_t num_base_inputs = GetRnnNumInputs(param);
  if (param.mode == rnn_enum::kLstm)
    TYPE_ASSIGN_CHECK(*in_type, quantized_rnn::kStateCell, mshadow::kFloat32);
  for (size_t i = num_base_inputs; i < num_inputs; ++i)
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);

  TYPE_ASSIGN_CHECK(*out_type, quantized_rnn::kOut, mshadow::kFloat32);
  if (param.state_outputs) {
    TYPE_ASSIGN_CHECK(*out_type, quantized_rnn::kStateOut, mshadow::kFloat32);
    if (param.mode == rnn_enum::kLstm)
      TYPE_ASSIGN_CHECK(*out_type, quantized_rnn::kStateCellOut, mshadow::kFloat32);
  }
  return true;
}

bool QuantizedRnnStorageType(const nnvm::NodeAttrs& attrs,
                             const int dev_mask,
                             DispatchMode* dispatch_mode,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  const uint32_t num_inputs  = QuantizedRnnNumInputs(attrs);
  const uint32_t num_outputs = QuantizedRnnNumOutputs(attrs);
  CHECK_EQ(in_attrs->size(), num_inputs);
  CHECK_EQ(out_attrs->size(), num_outputs);

#if MXNET_USE_ONEDNN == 1
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
#else
  *dispatch_mode = DispatchMode::kFCompute;

  for (auto& v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto& v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
#endif
}

void QuantizedRnnParamParser(nnvm::NodeAttrs* attrs) {
  RNNParam param;
  attrs->dict["quantized"] = "true";
  try {
    param.Init(attrs->dict, dmlc::parameter::kAllowUnknown);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

OpStatePtr CreateQuantizedRnnState(const nnvm::NodeAttrs& attrs,
                                   const Context ctx,
                                   const mxnet::ShapeVector& in_shapes,
                                   const std::vector<int>& in_types) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm) << "Quantized RNN operator only supports LSTM mode.";
  OpStatePtr state = OpStatePtr();
#if MXNET_USE_ONEDNN == 1
  const int data_type   = in_types[quantized_rnn::kData];
  const int weight_type = in_types[quantized_rnn::kParams];
  if (data_type == mshadow::kUint8 && weight_type == mshadow::kFloat32) {
    const mxnet::TShape& data_shape = in_shapes[quantized_rnn::kData];
    state =
        OpStatePtr::Create<DNNLQuantizedRnnOp>(attrs, data_shape[0], data_shape[1], data_shape[2]);
  }
#else
  LOG(FATAL) << "Quantized RNN operator relies on oneDNN library."
             << " Please build MXNet with USE_ONEDNN=ON to leverage this operator.";
#endif
  return state;
}

void QuantizedRnnForwardCPU(const OpStatePtr& state_ptr,
                            const OpContext& ctx,
                            const std::vector<TBlob>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& out_data) {
  LOG(FATAL) << "Quantized RNN operator relies on oneDNN library."
             << " Please build MXNet with USE_ONEDNN=ON to leverage this operator.";
}

#if MXNET_USE_ONEDNN == 1
void QuantizedRnnForwardCPUEx(const OpStatePtr& state_ptr,
                              const OpContext& ctx,
                              const std::vector<NDArray>& in_data,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& out_data) {
  DNNLQuantizedRnnOp& op = state_ptr.get_state<DNNLQuantizedRnnOp>();
  op.Forward(ctx, in_data, req, out_data);
}
#endif  // MXNET_USE_ONEDNN == 1

bool NeedAsymQuantizeRnnInput(const NodeAttrs& attrs, const size_t index_to_check) {
  bool need_asym_quantize = false;
  switch (index_to_check) {
    case rnn_enum::kData: {
      need_asym_quantize = true;
      break;
    }
    default: {
      need_asym_quantize = false;
    }
  }
  return need_asym_quantize;
}

bool AvoidRnnQuantizeInput(const NodeAttrs& attrs,
                           const size_t index_to_check,
                           const std::string quantize_granularity) {
  std::unordered_set<size_t> avoid_indexes;
  avoid_indexes.insert({quantized_rnn::kParams, quantized_rnn::kState, quantized_rnn::kStateCell});

  return avoid_indexes.count(index_to_check);
}

bool AvoidRnnDequantizeOutput(const NodeAttrs& attrs, const size_t index_to_check) {
  return true;
}

static std::vector<ResourceRequest> QuantizedRnnResourceEx(const NodeAttrs& attrs,
                                                           const int dev_mask,
                                                           const DispatchMode dispatch_mode) {
  std::vector<ResourceRequest> request;
  if (dev_mask == kGPU) {
#if MXNET_USE_CUDNN == 1
    LOG(FATAL) << "Currently, quantized RNN is not supported on the GPU platform.";
#endif
  } else {
#if MXNET_USE_ONEDNN == 1
    request.emplace_back(ResourceRequest::kTempSpace);
#endif
  }
  return request;
}

NNVM_REGISTER_OP(_contrib_quantized_rnn)
    .add_alias("_npx_contrib_quantized_rnn")
    .describe(R"code(RNN operator for input data type of uint8. The weight of each
gates is converted to int8, while bias is accumulated in type float32.
The hidden state and cell state are in type float32. For the input data, two more arguments
of type float32 must be provided representing the thresholds of quantizing argument from
data type float32 to uint8. The final outputs contain the recurrent result in float32.
It only supports quantization for Vanilla LSTM network.

.. Note::
    This operator only supports forward propagation. DO NOT use it in training.)code" ADD_FILELINE)
    .set_num_inputs(QuantizedRnnNumInputs)
    .set_num_outputs(QuantizedRnnNumOutputs)
    .set_attr_parser(QuantizedRnnParamParser)
    .set_attr<nnvm::FListInputNames>("FListInputNames", QuantizedRnnInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", QuantizedRnnOutputNames)
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedRnnShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedRnnType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedRnnStorageType)
    .set_attr<FCreateOpState>("FCreateOpState", CreateQuantizedRnnState)
    .set_attr<FStatefulCompute>("FStatefulCompute<cpu>", QuantizedRnnForwardCPU)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", QuantizedRnnForwardCPUEx)
#endif
    .set_attr<FResourceRequestEx>("FResourceRequestEx", QuantizedRnnResourceEx)
    .add_argument("data", "NDArray-or-Symbol", "Input data.")
    .add_argument("parameters", "NDArray-or-Symbol", "weight.")
    .add_argument("state", "NDArray-or-Symbol", "initial hidden state of the RNN")
    .add_argument("state_cell",
                  "NDArray-or-Symbol",
                  "initial cell state for LSTM networks (only for LSTM)")
    .add_argument("data_scale", "NDArray-or-Symbol", "quantization scale of data.")
    .add_argument("data_shift", "NDArray-or-Symbol", "quantization shift of data.")
    .add_arguments(RNNParam::__FIELDS__());

NNVM_REGISTER_OP(RNN)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) {
#if MXNET_USE_ONEDNN == 1
                              const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
                              if (param.mode != rnn_enum::kLstm)
                                LOG(INFO) << "Quantized RNN only supports LSTM mode.";
                              if (param.mode == rnn_enum::kLstm &&
                                  !param.projection_size.has_value()) {
                                return QuantizeType::kMust;
                              } else {
                                return QuantizeType::kNone;
                              }
#else
    LOG(INFO) << "Quantized RNN is not supported by this MXNet release. Please enable oneDNN to "
              << "use the feature.";
    return QuantizeType::kNone;
#endif  // MXNET_USE_ONEDNN == 1
                            })
    .set_attr<FQuantizedOp>("FQuantizedOp",
                            [](const NodeAttrs& attrs) {
                              nnvm::ObjectPtr node          = nnvm::Node::Create();
                              node->attrs.op                = Op::Get("_contrib_quantized_rnn");
                              node->attrs.name              = "quantized_" + attrs.name;
                              node->attrs.dict              = attrs.dict;
                              node->attrs.dict["quantized"] = "true";
                              if (node->op()->attr_parser != nullptr) {
                                node->op()->attr_parser(&(node->attrs));
                              }
                              return node;
                            })
    .set_attr<FNeedAsymQuantizeInput>("FNeedAsymQuantizeInput", NeedAsymQuantizeRnnInput)
    .set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", AvoidRnnQuantizeInput)
    .set_attr<FAvoidDequantizeOutput>("FAvoidDequantizeOutput", AvoidRnnDequantizeOutput);

}  // namespace op
}  // namespace mxnet
