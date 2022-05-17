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
 * \file rnn.cc
 * \brief
 * \author Sebastian Bodenstein
 */

#include <iterator>

#include "./rnn-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "./nn/dnnl/dnnl_rnn-inl.h"
#endif  // MXNET_USE_ONEDNN == 1

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(RNNParam);
static inline std::vector<std::string> ListRnnInputNames(const RNNParam& param) {
  // All RNNs start off with same 3 input arguments
  std::vector<std::string> arguments{"data", "parameters", "state"};

  // LSTMs also have an additional state_cell argument
  if (param.mode == rnn_enum::kLstm) {
    arguments.emplace_back("state_cell");
  }

  // All RNNs have option of additional sequence_length argument
  if (param.use_sequence_length) {
    arguments.emplace_back("sequence_length");
  }

  return arguments;
}

static inline std::vector<std::string> ListRnnOutputNames(const RNNParam& param) {
  std::vector<std::string> names{"output"};
  if (param.state_outputs) {
    names.emplace_back("state_output");
    if (param.mode == rnn_enum::kLstm)
      names.emplace_back("statecell_output");
  }
  return names;
}

static bool RNNShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape>* in_shape,
                     std::vector<TShape>* out_shape) {
  using namespace mshadow;
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);

  // Query param object to figure out what the expectd input arguments are
  std::vector<std::string> expected_arguments = ListRnnInputNames(param);

  CHECK_EQ(in_shape->size(), expected_arguments.size())
      << "Input shape mismatch. Expected " << expected_arguments.size()
      << " input parameters but got " << in_shape->size() << ".";

  const TShape& dshape = (*in_shape)[rnn_enum::kData];
  if (!mxnet::ndim_is_known(dshape))
    return false;
  CHECK_EQ(dshape.ndim(), 3U)
      << "Input data should be rank-3 tensor of dim [sequence length, batch size, input size]";
  // data: [sequence len, batch, input dimension]
  for (int i = 0; i < dshape.ndim(); i++) {
    CHECK_LT(dshape[i], INT32_MAX) << "ValueError: RNN does not support large"
                                   << "dimensions (>= 2^31).";
  }
  int batch_size    = dshape[1];
  int input_size    = dshape[2];
  int numDirections = param.bidirectional ? 2 : 1;
  int total_layers  = numDirections * param.num_layers;  // double for bidirectional
  int layer_size =
      (param.projection_size.has_value()) ? param.projection_size.value() : param.state_size;
  SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kState, Shape3(total_layers, batch_size, layer_size));
  if (param.mode == rnn_enum::kLstm) {
    SHAPE_ASSIGN_CHECK(
        *in_shape, rnn_enum::kStateCell, Shape3(total_layers, batch_size, param.state_size));
  }

  // calculate parameter vector length
  int param_size = GetRnnParamSize(param.num_layers,
                                   input_size,
                                   param.state_size,
                                   numDirections,
                                   param.mode,
                                   param.projection_size);
  SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kParams, Shape1(param_size));

  // Check on sequence_length shape if using
  if (param.use_sequence_length) {
    size_t seq_len_input_idx = rnn_enum::kSequenceLength;
    if (param.mode != rnn_enum::kLstm)
      --seq_len_input_idx;

    SHAPE_ASSIGN_CHECK(*in_shape, seq_len_input_idx, Shape1(batch_size));
  }

  out_shape->clear();
  // output: [sequence len, batch, output size]
  TShape oshape = dshape;
  if (param.projection_size.has_value()) {
    oshape[2] = numDirections * param.projection_size.value();
  } else {
    oshape[2] = numDirections * param.state_size;
  }
  out_shape->push_back(oshape);
  if (param.state_outputs) {
    // outStateShape: [layer_num, batch, state size]
    TShape outStateShape = dshape;
    outStateShape[0]     = total_layers;
    outStateShape[1]     = batch_size;
    if (param.projection_size.has_value()) {
      outStateShape[2] = param.projection_size.value();
    } else {
      outStateShape[2] = param.state_size;
    }
    out_shape->push_back(outStateShape);
    // Deal with lstm cell state
    if (param.mode == rnn_enum::kLstm) {
      TShape cellStateShape = dshape;
      cellStateShape[0]     = total_layers;
      cellStateShape[1]     = batch_size;
      cellStateShape[2]     = param.state_size;
      out_shape->push_back(cellStateShape);
    }
  }

  return true;
}

static bool RNNType(const nnvm::NodeAttrs& attrs,
                    std::vector<int>* in_type,
                    std::vector<int>* out_type) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);

  CHECK_EQ(in_type->size(), GetRnnNumInputs(param));

  size_t seq_len_input_idx = rnn_enum::kSequenceLength;
  if (param.mode != rnn_enum::kLstm)
    --seq_len_input_idx;

  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  std::vector<std::string> arguments = ListRnnInputNames(param);
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      TYPE_ASSIGN_CHECK(*in_type, i, dtype);
    } else {
      // If using sequence length argument, it has its own indexing type
      // All other input arguments must match the main data type
      if (!(param.use_sequence_length && i == seq_len_input_idx)) {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, arguments[i]);
      }
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  if (param.state_outputs) {
    out_type->push_back(dtype);
    // Deal with lstm cell state
    if (param.mode == rnn_enum::kLstm) {
      out_type->push_back(dtype);
    }
  }
  return true;
}

static std::vector<ResourceRequest> RNNResourceEx(const NodeAttrs& attrs,
                                                  const int dev_mask,
                                                  const DispatchMode dispatch_mode) {
  std::vector<ResourceRequest> request;
  if (dev_mask == kGPU) {
#if MXNET_USE_CUDNN == 1
    request.emplace_back(ResourceRequest::kTempSpace);
    request.emplace_back(ResourceRequest::kCuDNNDropoutDesc);
#endif
  } else {
    request.emplace_back(ResourceRequest::kRandom);
#if MXNET_USE_ONEDNN == 1
    request.emplace_back(ResourceRequest::kTempSpace);
#endif
  }
  return request;
}

#if MXNET_USE_ONEDNN == 1
inline static bool RNNStorageType(const nnvm::NodeAttrs& attrs,
                                  const int dev_mask,
                                  DispatchMode* dispatch_mode,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  const bool support_dnnl_rnn =
      !param.use_sequence_length && dmlc::GetEnv("MXNET_USE_ONEDNN_RNN", 1);
  return DNNLStorageType(attrs, dev_mask, support_dnnl_rnn, dispatch_mode, in_attrs, out_attrs);
}
#endif  // MXNET_USE_ONEDNN == 1

struct RNNGrad {
  const char* op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograd) const {
    const RNNParam& params = nnvm::get<RNNParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> heads{
        n->inputs[rnn_enum::kData], n->inputs[rnn_enum::kParams], n->inputs[rnn_enum::kState]};
    heads.emplace_back(n, rnn_enum::kOut, 0);
    heads.push_back(ograd[rnn_enum::kOut]);
    if (params.state_outputs) {
      heads.emplace_back(n, rnn_enum::kStateOut, 0);
      heads.push_back(ograd[rnn_enum::kStateOut]);
    }
    if (params.mode == rnn_enum::kLstm) {
      heads.push_back(n->inputs[rnn_enum::kStateCell]);
      if (params.state_outputs) {
        heads.emplace_back(n, rnn_enum::kStateCellOut, 0);
        heads.push_back(ograd[rnn_enum::kStateCellOut]);
      }
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

static OpStatePtr CreateRNNState(const nnvm::NodeAttrs& attrs,
                                 const Context ctx,
                                 const mxnet::ShapeVector& in_shapes,
                                 const std::vector<int>& in_types) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  OpStatePtr state      = OpStatePtr();
  int dtype             = in_types[rnn_enum::kData];
  int itype             = dtype;
  if (param.use_sequence_length) {
    size_t seq_len_input_idx = rnn_enum::kSequenceLength;
    if (param.mode != rnn_enum::kLstm) {
      seq_len_input_idx -= 1;
    }
    itype = in_types[seq_len_input_idx];
  }

#if MXNET_USE_ONEDNN == 1
  if (ctx.dev_type == kCPU && SupportDNNLRnn(param, in_types[rnn_enum::kData])) {
    const mxnet::TShape& data_shape = in_shapes[rnn_enum::kData];
    state = OpStatePtr::Create<DNNLRnnOp>(attrs, data_shape[0], data_shape[1], data_shape[2]);
    return state;
  }
#endif  // MXNET_USE_ONEDNN == 1

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    MSHADOW_TYPE_SWITCH(itype, IType, {
      if (ctx.dev_type == kGPU) {
        state = OpStatePtr::Create<RNNOp<gpu, DType, IType>>(param, ctx);
      } else {
        state = OpStatePtr::Create<RNNOp<cpu, DType, IType>>(param, ctx);
      }
    });
  });
  return state;
}

#if MXNET_USE_ONEDNN == 1
static void RNNStatefulComputeExCPU(const OpStatePtr& state_ptr,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  if (SupportDNNLRnn(inputs[rnn_enum::kData].dtype())) {
    DNNLRnnOp& op = state_ptr.get_state<DNNLRnnOp>();
    op.Forward(ctx, inputs, req, outputs);
  } else {
    FallBackCompute(RNNStatefulCompute<cpu>, state_ptr, ctx, inputs, req, outputs);
  }
}

static void RNNStatefulGradComputeExCPU(const OpStatePtr& state_ptr,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  if (SupportDNNLRnn(inputs[rnn_enum::kData].dtype())) {
    DNNLRnnOp& op = state_ptr.get_state<DNNLRnnOp>();
    op.Backward(ctx, inputs, req, outputs);
  } else {
    FallBackCompute(RNNStatefulGradCompute<cpu>, state_ptr, ctx, inputs, req, outputs);
  }
}
#endif  // MXNET_USE_ONEDNN == 1

NNVM_REGISTER_OP(RNN)
    .add_alias("_npx_rnn")
    .describe(
        R"code(Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
implemented, with both multi-layer and bidirectional support.

When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

**Vanilla RNN**

Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported:
ReLU and Tanh.

With ReLU activation function:

.. math::
    h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

With Tanh activtion function:

.. math::
    h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

Reference paper: Finding structure in time - Elman, 1988.
https://axon.cs.byu.edu/~martinez/classes/678/Papers/Elman_time.pdf

**LSTM**

Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf

.. math::
  \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array}

With the projection size being set, LSTM could use the projection feature to reduce the parameters
size and give some speedups without significant damage to the accuracy.

Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech
Recognition - Sak et al. 2014. https://arxiv.org/abs/1402.1128

.. math::
  \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{o} + W_{ro} r_{(t-1)} + b_{ro}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            r_t = W_{hr} h_t
            \end{array}

**GRU**

Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078

The definition of GRU here is slightly different from paper but compatible with CUDNN.

.. math::
  \begin{array}{ll}
            r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
            \end{array}
)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<RNNParam>)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
      return GetRnnNumInputs(params);
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
      //  kOut
      int num_outputs = 1;
      if (params.state_outputs) {
        // kOut, kStateOut, kStateCellOut
        num_outputs = (params.mode == rnn_enum::kLstm) ? 3 : 2;
      }

      return num_outputs;
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
                                       return ListRnnInputNames(params);
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
                                        return ListRnnOutputNames(params);
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", RNNShape)
    .set_attr<nnvm::FInferType>("FInferType", RNNType)
    .set_attr<FCreateOpState>("FCreateOpState", CreateRNNState)
    .set_attr<FStatefulCompute>("FStatefulCompute<cpu>", RNNStatefulCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", RNNStorageType)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", RNNStatefulComputeExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", RNNGrad{"_backward_RNN"})
    .set_attr<FResourceRequestEx>("FResourceRequestEx", RNNResourceEx)
    .add_argument("data", "NDArray-or-Symbol", "Input data to RNN")
    .add_argument("parameters",
                  "NDArray-or-Symbol",
                  "Vector of all RNN trainable parameters concatenated")
    .add_argument("state", "NDArray-or-Symbol", "initial hidden state of the RNN")
    .add_argument("state_cell",
                  "NDArray-or-Symbol",
                  "initial cell state for LSTM networks (only for LSTM)")
    .add_argument("sequence_length",
                  "NDArray-or-Symbol",
                  "Vector of valid sequence lengths for each element in batch. (Only used if"
                  " use_sequence_length kwarg is True)")
    .add_arguments(RNNParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_RNN)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
      int ret                = 5;
      if (params.state_outputs) {
        ret += 2;
      }
      if (params.mode == rnn_enum::kLstm) {
        ++ret;
        if (params.state_outputs) {
          ret += 2;
        }
      }
      return ret;
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const RNNParam& params = nnvm::get<RNNParam>(attrs.parsed);
      return GetRnnNumInputs(params);
    })
    .set_attr_parser(ParamParser<RNNParam>)
    .set_attr<bool>("TIsLayerOpBackward", true)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FStatefulCompute>("FStatefulCompute<cpu>", RNNStatefulGradCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", RNNStorageType)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", RNNStatefulGradComputeExCPU)
#endif
    .set_attr<FResourceRequestEx>("FResourceRequestEx", RNNResourceEx);
}  // namespace op
}  // namespace mxnet
