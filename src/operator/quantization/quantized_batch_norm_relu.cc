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
 * \file quantized_batch_norm_relu.cc
 * \author Hanna Jarlaczyńska, hanna.jarlaczynska@intel.com
 */
#include <mxnet/op_attr_types.h>
#include "operator/nn/batch_norm-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_batch_norm-inl.h"
#endif

namespace mxnet {
namespace op {

bool QuantizedBatchNormWithReLUShape(const nnvm::NodeAttrs& attrs,
                                     mxnet::ShapeVector* in_shape,
                                     mxnet::ShapeVector* out_shape) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 7U)
      << "Input:[data, gamma, beta, moving_mean, moving_var, min_data, max_data]";
  CHECK_EQ(out_shape->size(), 3U);

  const mxnet::TShape& dshape = in_shape->at(batchnorm::kData);
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }
  const int channelAxis = batchnorm::GetRealAxis(dshape, param.axis);
  CHECK(channelAxis >= 0 && channelAxis < dshape.ndim())
      << "Channel axis out of range: " << param.axis;
  const int channelCount = dshape[channelAxis];

  SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape(Shape1(channelCount)))  // gamma,beta
  SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape(Shape1(channelCount)))
  SHAPE_ASSIGN_CHECK(*in_shape, 3, mxnet::TShape(Shape1(channelCount)));  // moving_mean, moving_var
  SHAPE_ASSIGN_CHECK(*in_shape, 4, mxnet::TShape(Shape1(channelCount)))
  SHAPE_ASSIGN_CHECK(*in_shape, 5, mxnet::TShape(1, 1));  // min_data, max_data
  SHAPE_ASSIGN_CHECK(*in_shape, 6, mxnet::TShape(1, 1));

  SHAPE_ASSIGN_CHECK(*out_shape, 0, dshape);
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));  // min_output, max_output
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));
  return true;
}

bool QuantizedBatchNormWithReLUType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_type,
                                    std::vector<int>* out_type) {
  using namespace mshadow;
  CHECK_EQ(in_type->size(), 7U);
  CHECK_EQ(out_type->size(), 3U);

#if MXNET_USE_ONEDNN == 1
  CHECK(in_type->at(0) == mshadow::kInt8 || in_type->at(0) == mshadow::kUint8)
      << "QuantizedBatchNorm with oneDNN backend only supports int8/uint8 input, while "
      << in_type->at(0) << " is given.";
#else
  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
#endif
  for (size_t i = 1; i < 7; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);

  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_batch_norm_relu)
    .describe(R"code(BatchNorm with ReLU operator for input and output data type of int8.
The input and output data comes with min and max thresholds for quantizing
the float32 data into int8.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.
)code" ADD_FILELINE)
    .set_num_inputs(7)
    .set_num_outputs(3)
    .set_attr_parser(ParamParser<BatchNormParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{
              "data", "gamma", "beta", "moving_mean", "moving_var", "min_data", "max_data"};
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"output", "min_output", "max_output"};
        })
    .set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                   [](const nnvm::NodeAttrs& attrs) {
                                     return std::vector<uint32_t>{3, 4};
                                   })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedBatchNormWithReLUShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedBatchNormWithReLUType)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return false; })
    .set_attr<FNeedCalibrateInput>("FNeedCalibrateOutput",
                                   [](const NodeAttrs& attrs) { return std::vector<int>{0}; })
    .add_argument("data", "NDArray-or-Symbol", "Input data.")
    .add_argument("gamma", "NDArray-or-Symbol", "gamma.")
    .add_argument("beta", "NDArray-or-Symbol", "beta.")
    .add_argument("moving_mean", "NDArray-or-Symbol", "moving_mean.")
    .add_argument("moving_var", "NDArray-or-Symbol", "moving_var.")
    .add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
    .add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
    .add_arguments(BatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(_sg_onednn_batch_norm)
    .set_attr<FQuantizedOp>("FQuantizedOp",
                            [](const NodeAttrs& attrs) {
                              nnvm::ObjectPtr node = nnvm::Node::Create();
                              node->attrs.op       = Op::Get("_contrib_quantized_batch_norm_relu");
                              node->attrs.name     = "quantized_" + attrs.name;
                              node->attrs.dict     = attrs.dict;
                              if (node->op()->attr_parser != nullptr) {
                                node->op()->attr_parser(&(node->attrs));
                              }
                              return node;
                            })
    .set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput",
                                   [](const NodeAttrs& attrs,
                                      const size_t index,
                                      const std::string quantize_granularity) {
                                     return (index != 0);
                                   });

}  // namespace op
}  // namespace mxnet
