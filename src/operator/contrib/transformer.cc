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
 *  Copyright (c) 2018 by Contributors
 * \file transformer.cc
 * \brief CPU implementation of the operators used in Transformer
 */
#include <mxnet/base.h>
#include "./transformer-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InterleavedMatMulParam);

static bool InterleavedMatMulSelfAttQKShape(const NodeAttrs& attrs,
                                            mxnet::ShapeVector* in_shape,
                                            mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U) << "Input:[queries_keys_values] currently have, "
                                 << in_shape->size() << " inputs";
  auto qkv_shape = in_shape->at(0);
  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  out_shape->resize(1);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({params.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));
  return true;
}

static bool InterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  CHECK_EQ(in_shape->size(), 2U) << "Input:[queries_keys_values, attention] currently have, "
                                 << in_shape->size() << " inputs";
  auto qkv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-3*proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  CHECK_EQ(att_shape.ndim(), 3U)
    << "Input attention should be 3D in batch-seq_length-seq_length, "
    << "currently is: " << att_shape.ndim() << "D";
  CHECK_EQ(qkv_shape[0], att_shape[1])
    << "queries_keys_values.shape[0] and attention.shape[1] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[1];
  CHECK_EQ(qkv_shape[0], att_shape[2])
    << "queries_keys_values.shape[0] and attention.shape[2] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[2];
  CHECK_EQ(qkv_shape[2] % 3, 0)
    << "queries_keys_values.shape[2] should be a multiple of 3, "
    << "currently is " << qkv_shape[2];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
  return true;
}

static bool InterleavedMatMulEncDecQKShape(const NodeAttrs& attrs,
                                           mxnet::ShapeVector* in_shape,
                                           mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input:[queries, keys_values], currently have "
                                 << in_shape->size() << " inputs";
  auto q_shape = in_shape->at(0);
  auto kv_shape = in_shape->at(1);
  CHECK_EQ(q_shape.ndim(), 3U) << "Input queries should be 3D in seq_length-batch-proj_dim, "
                               << "currently is " << q_shape.ndim() << "D";
  CHECK_EQ(kv_shape.ndim(), 3U) << "Input queries should be 3D in seq_length-batch-2*proj_dim, "
                                << "currently is " << kv_shape.ndim() << "D";
  CHECK_EQ(q_shape[2] * 2, kv_shape[2])
    << "keys_values.shape[2] should be equal to queries.shape[2] * 2, "
    << "currently are: " << kv_shape[2] << " and " << q_shape[2];
  CHECK_EQ(q_shape[1], kv_shape[1])
    << "queries.shape[1] should be equal to keys_values.shape[1], "
    << "currently are: " << q_shape[1] << " and " << kv_shape[1];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({q_shape[1] * params.heads, q_shape[0], kv_shape[0]}));
  return true;
}

static bool InterleavedMatMulEncDecValAttShape(const NodeAttrs& attrs,
                                               mxnet::ShapeVector* in_shape,
                                               mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input: [keys_values, attention], currently have "
                                 << in_shape->size() << " inputs";
  auto kv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(kv_shape.ndim(), 3U)
    << "Input keys_values should be 3D in seq_length-batch-2*proj_dim, "
    << "currently is " << kv_shape.ndim() << "D";
  CHECK_EQ(att_shape.ndim(), 3U)
    << "Input attention should be 3D in batch-seq_length-seq_length, "
    << "currently is " << att_shape.ndim() << "D";
  CHECK_EQ(kv_shape[0], att_shape[2])
    << "keys_values.shape[0] should be equal to attention.shape[2], currently are "
    << kv_shape[0] << " and " << att_shape[2];
  CHECK_EQ(kv_shape[1] * params.heads, att_shape[0]) << "attention.shape[0] "
    << "should be equal to keys_values.shape[1] * heads, currently are: "
    << att_shape[2] << " and " << kv_shape[1];
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({att_shape[1], kv_shape[1], kv_shape[2] / 2}));
  return true;
}

NNVM_REGISTER_OP(_contrib_interleaved_matmul_selfatt_qk)
.describe(R"code(Compute the matrix multiplication between the projections of
queries and keys in multihead attention use as self attention.

the input must be a single tensor of interleaved projections
of queries, keys and values following the layout:
(seq_length, batch_size, num_heads * head_dim * 3)

the equivalent code would be:
tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
q_proj = mx.nd.transpose(tmp[:,:,:,0,:], axes=(1, 2, 0, 3))
q_proj = mx.nd.reshape(q_proj, shape=(-1, 0, 0), reverse=True)
q_proj = mx.nd.contrib.div_sqrt_dim(q_proj)
k_proj = mx.nd.transpose(tmp[:,:,:,1,:], axes=(1, 2, 0, 3))
k_proj = mx.nd.reshap(k_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(q_proj, k_proj, transpose_b=True)

This Op is GPU only
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseIn{"_backward_interleaved_matmul_selfatt_qk"})
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_qk)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);

NNVM_REGISTER_OP(_contrib_interleaved_matmul_selfatt_valatt)
.describe(R"code(Compute the matrix multiplication between the projections of
values and the attention weights in multihead attention use as self attention.

the inputs must be a tensor of interleaved projections
of queries, keys and values following the layout:
(seq_length, batch_size, num_heads * head_dim * 3)

and the attention weights following the layout:
(batch_size, seq_length, seq_length)

the equivalent code would be:
tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
v_proj = mx.nd.transpose(tmp[:,:,:,2,:], axes=(1, 2, 0, 3))
v_proj = mx.nd.reshape(v_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(attention, v_proj, transpose_b=True)
output = mx.nd.reshape(output, shape=(-1, num_heads, 0, 0), reverse=True)
output = mx.nd.transpose(output, axes=(0, 2, 1, 3))
output = mx.nd.reshape(output, shape=(0, 0, -1))

This Op is GPU only
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulSelfAttValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseIn{"_backward_interleaved_matmul_selfatt_valatt"})
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_valatt)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);

NNVM_REGISTER_OP(_contrib_interleaved_matmul_encdec_qk)
.describe(R"code(Compute the matrix multiplication between the projections of
queries and keys in multihead attention use as encoder-decoder.

the inputs must be a tensor of projections of queries following the layout:
(seq_length, batch_size, num_heads * head_dim)

and a tensor of interleaved projections of values and keys following the layout:
(seq_length, batch_size, num_heads * head_dim * 2)

the equivalent code would be:
q_proj = mx.nd.transpose(queries, axes=(1, 2, 0, 3))
q_proj = mx.nd.reshape(q_proj, shape=(-1, 0, 0), reverse=True)
q_proj = mx.nd.contrib.div_sqrt_dim(q_proj)
tmp = mx.nd.reshape(keys_values, shape=(0, 0, num_heads, 2, -1))
k_proj = mx.nd.transpose(tmp[:,:,:,0,:], axes=(1, 2, 0, 3))
k_proj = mx.nd.reshap(k_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(q_proj, k_proj, transpose_b=True)

This Op is GPU only
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries", "keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulEncDecQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
    ElemwiseGradUseIn{"_backward_interleaved_matmul_encdec_qk"})
.add_argument("queries", "NDArray-or-Symbol", "Queries")
.add_argument("keys_values", "NDArray-or-Symbol", "Keys and values interleaved")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_encdec_qk)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);

NNVM_REGISTER_OP(_contrib_interleaved_matmul_encdec_valatt)
.describe(R"code(Compute the matrix multiplication between the projections of
values and the attention weights in multihead attention use as encoder-decoder.

the inputs must be a tensor of interleaved projections of
keys and values following the layout:
(seq_length, batch_size, num_heads * head_dim * 2)

and the attention weights following the layout:
(batch_size, seq_length, seq_length)

the equivalent code would be:

tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
v_proj = mx.nd.transpose(tmp[:,:,:,1,:], axes=(1, 2, 0, 3))
v_proj = mx.nd.reshape(v_proj, shape=(-1, 0, 0), reverse=True)
output = mx.nd.batch_dot(attention, v_proj, transpose_b=True)
output = mx.nd.reshape(output, shape=(-1, num_heads, 0, 0), reverse=True)
output = mx.nd.transpose(output, axes=(0, 2, 1, 3))
output = mx.nd.reshape(output, shape=(0, 0, -1))

This Op is GPU only
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulEncDecValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
    ElemwiseGradUseIn{"_backward_interleaved_matmul_encdec_valatt"})
.add_argument("keys_values", "NDArray-or-Symbol", "Keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_encdec_valatt)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);


// relu
MXNET_OPERATOR_REGISTER_UNARY(_contrib_div_sqrt_dim)
.describe(R"code(Rescale the input by the square root of the channel dimension.

   out = data / sqrt(data.shape[-1])

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", DivSqrtDimForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_div_sqrt_dim"});

}  // namespace op
}  // namespace mxnet
