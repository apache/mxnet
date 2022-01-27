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
 * \file quantized_reshape.cc
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#include "quantized_reshape-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuantizedReshapeParam);

void QuantizedReshapeCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);
  CHECK_EQ(req.size(), 3U);

  if (req[0] != kWriteInplace)
    UnaryOp::IdentityCompute<cpu>(attrs, ctx, inputs, req, outputs);

  *outputs[1].dptr<float>() = *inputs[1].dptr<float>();
  *outputs[2].dptr<float>() = *inputs[2].dptr<float>();
}

NNVM_REGISTER_OP(_contrib_quantized_reshape)
    .add_alias("_npx_quantized_reshape")
    .set_num_inputs(3)
    .set_num_outputs(3)
    .set_attr_parser(ParamParser<QuantizedReshapeParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"data", "min_data", "max_data"};
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"output", "min_output", "max_output"};
        })
    .set_attr<nnvm::FInplaceOption>(
        "FInplaceOption",
        [](const NodeAttrs& attrs) {
          return std::vector<std::pair<int, int> >{{0, 0}, {1, 1}, {2, 2}};
        })
    .set_attr<FCompute>("FCompute<cpu>", QuantizedReshapeCompute)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedReshapeInferShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedReshapeType)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kSupport; })
    .add_argument("data", "NDArray-or-Symbol", "Array to be reshaped.")
    .add_argument("min_data",
                  "NDArray-or-Symbol",
                  "The minimum scalar value "
                  "possibly produced for the data")
    .add_argument("max_data",
                  "NDArray-or-Symbol",
                  "The maximum scalar value "
                  "possibly produced for the data")
    .add_arguments(QuantizedReshapeParam::__FIELDS__());

template <bool is_numpy_op>
nnvm::ObjectPtr QuantizedReshapeNode(const NodeAttrs& attrs) {
  QuantizedReshapeParam param;
  if (is_numpy_op) {
    const NumpyXReshapeParam& _param = nnvm::get<NumpyXReshapeParam>(attrs.parsed);
    param.newshape                   = _param.newshape;
    param.reverse                    = _param.reverse;
    param.order                      = _param.order;
    param.keep_highest               = false;
    param.is_numpy_op                = true;
  } else {
    const ReshapeParam& _param = nnvm::get<ReshapeParam>(attrs.parsed);
    param.shape                = _param.shape;
    param.keep_highest         = _param.keep_highest;
    param.reverse              = _param.reverse;
    param.is_numpy_op          = false;
  }

  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op       = Op::Get("_contrib_quantized_reshape");
  node->attrs.name     = "quantized_" + attrs.name;
  param.SetAttrDict(&(node->attrs.dict));
  if (node->op() != nullptr && node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
}

NNVM_REGISTER_OP(_npx_reshape).set_attr<FQuantizedOp>("FQuantizedOp", QuantizedReshapeNode<true>);

NNVM_REGISTER_OP(Reshape).set_attr<FQuantizedOp>("FQuantizedOp", QuantizedReshapeNode<false>);

}  // namespace op
}  // namespace mxnet
