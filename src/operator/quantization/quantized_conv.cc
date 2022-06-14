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
 * \file quantized_conv.cc
 * \brief
 * \author Ziheng Jiang, Jun Wu
 */
#include "../nn/convolution-inl.h"

namespace mxnet {
namespace op {

bool QuantizedConvShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_shape,
                        mxnet::ShapeVector* out_shape) {
  using namespace mshadow;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  CHECK_EQ(param.num_group, 1U) << "quantized_conv only supports num_group=1 for now";
  CHECK_EQ(in_shape->size(), param.no_bias ? 6U : 9U);
  CHECK_EQ(out_shape->size(), 3U);
  if (param.layout.has_value()) {
#if MXNET_USE_ONEDNN == 1
    CHECK(param.layout.value() == mshadow::kNCHW || param.layout.value() == mshadow::kNCDHW)
        << "oneDNN quantized_conv only supports NCHW and NCDHW for now";
#else
    CHECK_EQ(param.layout.value(), mshadow::kNCHW) << "quantized_conv only supports NCHW for now";
#endif
  }

  const mxnet::TShape& dshape = in_shape->at(0);
  const int data_ndims        = dshape.ndim();
  const int kernel_ndims      = param.kernel.ndim();
  if (data_ndims == 0U)
    return false;

#if MXNET_USE_ONEDNN == 1
  CHECK(kernel_ndims == 2U || kernel_ndims == 3U)
      << "oneDNN quantized_conv only supports 2d and 3d kernel for now";
  CHECK(data_ndims == 4U || data_ndims == 5U)
      << "oneDNN quantized_conv only supports 4d and 5d layout for now";
#else
  CHECK_EQ(kernel_ndims, 2U) << "quantized_conv only supports 2D convolution for now";
  CHECK(param.dilate.ndim() == 0U || param.dilate.Size() == 1U)
      << "quantized_conv only supports dilation=1 for all dimensions";
  CHECK_EQ(data_ndims, 4U);
  CHECK_EQ(dshape[1] % 4, 0U) << "for 8bit cudnn conv, the number of channel must be multiple of 4";
  CHECK_EQ(param.num_filter % 4, 0U)
      << "for 8bit cudnn conv, the number of channel must be multiple of 4";
#endif

  auto AddPad = [](index_t dsize, index_t pad) { return dsize + 2 * pad; };
  const int D = (data_ndims == 5) ? 2 : 1;
  const int N = 0, H = D + 1, W = D + 2, C = 1;

  if (data_ndims == 4) {
    // conv 2d
    mxnet::TShape wshape(data_ndims, 0);
    wshape[N] = param.num_filter;
    wshape[H] = param.kernel[0];
    wshape[W] = param.kernel[1];
    wshape[C] = dshape[C];
    SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);

    mxnet::TShape oshape{1, 1, 1, 1};
    oshape[N] = dshape[N];
    oshape[C] = wshape[N];

    const index_t dilated_ksize_y = param.DilatedKernelSize(0);
    const index_t dilated_ksize_x = param.DilatedKernelSize(1);
    oshape[H] = (AddPad(dshape[H], param.pad[0]) - dilated_ksize_y) / param.stride[0] + 1;
    oshape[W] = (AddPad(dshape[W], param.pad[1]) - dilated_ksize_x) / param.stride[1] + 1;

    SHAPE_ASSIGN_CHECK(*out_shape, 0, oshape);
    SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));
    SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));
#if MXNET_USE_ONEDNN == 1
  } else {
    // conv 3d
    mxnet::TShape wshape(data_ndims, 0);
    wshape[N] = param.num_filter;
    wshape[D] = param.kernel[0];
    wshape[H] = param.kernel[1];
    wshape[W] = param.kernel[2];
    wshape[C] = dshape[C];
    SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);

    mxnet::TShape oshape{1, 1, 1, 1, 1};
    oshape[N] = dshape[N];
    oshape[C] = wshape[N];

    const index_t dilated_ksize_d = param.DilatedKernelSize(0);
    const index_t dilated_ksize_y = param.DilatedKernelSize(1);
    const index_t dilated_ksize_x = param.DilatedKernelSize(2);
    oshape[D] = (AddPad(dshape[D], param.pad[0]) - dilated_ksize_d) / param.stride[0] + 1;
    oshape[H] = (AddPad(dshape[H], param.pad[1]) - dilated_ksize_y) / param.stride[1] + 1;
    oshape[W] = (AddPad(dshape[W], param.pad[2]) - dilated_ksize_x) / param.stride[2] + 1;

    SHAPE_ASSIGN_CHECK(*out_shape, 0, oshape);
    SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));
    SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));
#endif
  }

  const int start = param.no_bias ? 2 : 3;
  const int end   = param.no_bias ? 6 : 9;
  for (int i = start; i < end; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape{1});
  }
  if (!param.no_bias) {
    SHAPE_ASSIGN_CHECK(*in_shape, 2, Shape1(param.num_filter));
  }

  return true;
}

bool QuantizedConvType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_type,
                       std::vector<int>* out_type) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.no_bias ? 6U : 9U);
  CHECK_EQ(out_type->size(), 3U);
#ifndef MXNET_USE_ONEDNN
  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
#endif
  TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kInt8);
  if (!param.no_bias) {
    TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kInt8);
  }
  const size_t start = param.no_bias ? 2 : 3;
  const size_t end   = param.no_bias ? 6 : 9;
  for (size_t i = start; i < end; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

bool QuantizedConvStorageType(const nnvm::NodeAttrs& attrs,
                              const int dev_mask,
                              DispatchMode* dispatch_mode,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_ONEDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif

  (*out_attrs)[0] = kDefaultStorage;
  (*out_attrs)[1] = kDefaultStorage;
  (*out_attrs)[2] = kDefaultStorage;
  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
    .add_alias("_npx_quantized_conv")
    .describe(R"code(Convolution operator for input, weight and bias data type of int8,
and accumulates in type int32 for the output. For each argument, two more arguments of type
float32 must be provided representing the thresholds of quantizing argument from data
type float32 to int8. The final outputs contain the convolution result in int32, and min
and max thresholds representing the threholds for quantizing the float32 output into int32.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
      return param.no_bias ? 6 : 9;
    })
    .set_num_outputs(3)
    .set_attr_parser(ConvolutionParamParser)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
          if (param.no_bias) {
            return std::vector<std::string>{
                "data", "weight", "min_data", "max_data", "min_weight", "max_weight"};
          } else {
            return std::vector<std::string>{"data",
                                            "weight",
                                            "bias",
                                            "min_data",
                                            "max_data",
                                            "min_weight",
                                            "max_weight",
                                            "min_bias",
                                            "max_bias"};
          }
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"output", "min_output", "max_output"};
        })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedConvShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedConvType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedConvStorageType)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>(1,
                                                                      ResourceRequest::kTempSpace);
                                })
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .add_argument("data", "NDArray-or-Symbol", "Input data.")
    .add_argument("weight", "NDArray-or-Symbol", "weight.")
    .add_argument("bias", "NDArray-or-Symbol", "bias.")
    .add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
    .add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
    .add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
    .add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
    .add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
    .add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
    .add_arguments(ConvolutionParam::__FIELDS__());

NNVM_REGISTER_OP(Convolution)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
      nnvm::ObjectPtr node = nnvm::Node::Create();
      node->attrs.op       = Op::Get("_contrib_quantized_conv");
      node->attrs.name     = "quantized_" + attrs.name;
      node->attrs.dict     = attrs.dict;
      if (node->op()->attr_parser != nullptr) {
        node->op()->attr_parser(&(node->attrs));
      }
      return node;
    });

}  // namespace op
}  // namespace mxnet
