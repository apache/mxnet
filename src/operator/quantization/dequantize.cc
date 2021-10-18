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
 * \file dequantize.cc
 * \brief
 */
#include "./dequantize-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "./dnnl/dnnl_dequantize-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DequantizeParam);

bool DequantizeStorageType(const nnvm::NodeAttrs& attrs,
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
  return true;
}

static OpStatePtr CreateDequantizeState(const nnvm::NodeAttrs& attrs,
                                        Context ctx,
                                        const std::vector<TShape>& in_shapes,
                                        const std::vector<int>& in_types) {
  OpStatePtr state;
  if (ctx.dev_type == kGPU) {
    state = OpStatePtr::Create<DequantizeOperator<gpu>>(attrs);
  } else {
#if MXNET_USE_ONEDNN == 1
    state = OpStatePtr::Create<SgDNNLDequantizeOperator>(attrs);
#else
    state = OpStatePtr::Create<DequantizeOperator<cpu>>(attrs);
#endif
  }
  return state;
}

NNVM_REGISTER_OP(_contrib_dequantize)
    .describe(R"code(Dequantize the input tensor into a float tensor.
min_range and max_range are scalar floats that specify the range for
the output data.

When input data type is `uint8`, the output is calculated using the following equation:

`out[i] = in[i] * (max_range - min_range) / 255.0`,

When input data type is `int8`, the output is calculate using the following equation
by keep zero centered for the quantized value:

`out[i] = in[i] * MaxAbs(min_range, max_range) / 127.0`,

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.
)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<DequantizeParam>)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"data", "min_range", "max_range"};
        })
    .set_attr<mxnet::FInferShape>("FInferShape", DequantizeShape)
    .set_attr<nnvm::FInferType>("FInferType", DequantizeType)
    .set_attr<FInferStorageType>("FInferStorageType", DequantizeStorageType)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FCreateOpState>("FCreateOpState", CreateDequantizeState)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLDequantizeForward)
#endif
    .set_attr<FStatefulCompute>("FStatefulCompute<cpu>", DequantizeForward<cpu>)
    .add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `uint8`")
    .add_argument("min_range",
                  "NDArray-or-Symbol",
                  "The minimum scalar value "
                  "possibly produced for the input in float32")
    .add_argument("max_range",
                  "NDArray-or-Symbol",
                  "The maximum scalar value "
                  "possibly produced for the input in float32")
    .add_arguments(DequantizeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
